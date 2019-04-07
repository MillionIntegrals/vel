import gym
import itertools as it
import torch
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator, RlModel
from vel.rl.modules.deterministic_action_head import DeterministicActionHead
from vel.rl.modules.deterministic_critic_head import DeterministicCriticHead


class DeterministicPolicyEvaluator(Evaluator):
    """ Evaluator for DeterministicPolicyModel """

    def __init__(self, model: 'DeterministicPolicyModel', rollout: Rollout):
        super().__init__(rollout)

        self.model = model

    @Evaluator.provides('model:values_next')
    def model_estimated_values_next(self):
        """ Estimate state-value of the transition next state """
        observations = self.get('rollout:observations_next')
        action, value = self.model(observations)
        return value

    @Evaluator.provides('model:actions')
    def model_actions(self):
        """ Estimate state-value of the transition next state """
        observations = self.get('rollout:observations')
        model_action = self.model.action(observations)
        return model_action

    @Evaluator.provides('model:model_action:q')
    def model_model_action_q(self):
        observations = self.get('rollout:observations')
        model_actions = self.get('model:actions')
        return self.model.value(observations, model_actions)

    @Evaluator.provides('model:action:q')
    def model_action_q(self):
        observations = self.get('rollout:observations')
        rollout_actions = self.get('rollout:actions')
        return self.model.value(observations, rollout_actions)


class DeterministicPolicyModel(RlModel):
    """ Deterministic Policy Gradient - model """

    def __init__(self, input_block: BackboneModel, policy_backbone: LinearBackboneModel,
                 value_backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = DeterministicActionHead(self.policy_backbone.output_dim, action_space)
        self.critic_head = DeterministicCriticHead(self.value_backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()
        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()
        self.action_head.reset_weights()
        self.critic_head.reset_weights()

    def forward(self, observations, input_actions=None):
        """ Calculate model outputs """
        observations = self.input_block(observations)

        if input_actions is not None:
            actions = input_actions

            value_input = torch.cat([observations, actions], dim=1)
            value_hidden = self.value_backbone(value_input)

            values = self.critic_head(value_hidden)
        else:
            policy_hidden = self.policy_backbone(observations)
            actions = self.action_head(policy_hidden)

            value_input = torch.cat([observations, actions], dim=1)
            value_hidden = self.value_backbone(value_input)

            values = self.critic_head(value_hidden)

        return actions, values

    def policy_parameters(self):
        """ Parameters of policy """
        return it.chain(self.policy_backbone.parameters(), self.action_head.parameters())

    def value_parameters(self):
        """ Parameters of policy """
        return it.chain(self.value_backbone.parameters(), self.critic_head.parameters())

    def get_layer_groups(self):
        """ Return layers grouped """
        return [
            [self.policy_backbone, self.action_head],
            [self.value_backbone, [y for (x, y) in self.critic_head.named_parameters() if x.endswith('bias')]],
            # OpenAI regularizes only weight on the last layer. I'm just replicating that
            [[y for (x, y) in self.critic_head.named_parameters() if x.endswith('weight')]]
        ]

    def step(self, observations):
        """ Select actions based on model's output """
        action, value = self(observations)

        return {
            'actions': action,
            'values': value
        }

    def value(self, observation, input_actions=None):
        """ Calculate value for given state """
        action, value = self(observation, input_actions)
        return value

    def action(self, observations):
        """ Calculate value for given state """
        observations = self.input_block(observations)
        policy_hidden = self.policy_backbone(observations)
        action = self.action_head(policy_hidden)
        return action

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return DeterministicPolicyEvaluator(self, rollout)


class DeterministicPolicyModelFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: ModelFactory, policy_backbone: ModelFactory, value_backbone: ModelFactory):
        self.input_block = input_block
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return DeterministicPolicyModel(
            input_block=input_block,
            policy_backbone=policy_backbone,
            value_backbone=value_backbone,
            action_space=extra_args['action_space'],
        )


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory,
           input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return DeterministicPolicyModelFactory(
        input_block=input_block, policy_backbone=policy_backbone, value_backbone=value_backbone
    )
