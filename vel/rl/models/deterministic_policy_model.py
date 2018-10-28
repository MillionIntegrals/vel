import gym
import itertools as it

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.modules.deterministic_action_head import DeterministicActionHead
from vel.rl.modules.deterministic_critic_head import DeterministicCriticHead


class DeterministicPolicyEvaluator(Evaluator):
    """ Evaluator for DeterministicPolicyModel """

    def __init__(self, model: 'DeterministicPolicyModel', rollout: Rollout):
        super().__init__(rollout)

        self.model = model

    @Evaluator.provides('model:estimated_values_next')
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


class DeterministicPolicyModel(Model):
    """ Deterministic Policy Gradient - model """

    def __init__(self, policy_backbone: LinearBackboneModel, value_backbone: LinearBackboneModel,
                 action_space: gym.Space, critic_hidden_dim: int=64,
                 critic_normalization='layer', critic_activation='relu'):
        super().__init__()

        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = DeterministicActionHead(self.policy_backbone.output_dim, action_space)
        self.critic_head = DeterministicCriticHead(
            self.value_backbone.output_dim, action_space,
            hidden_dim=critic_hidden_dim, normalization=critic_normalization, activation=critic_activation
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()
        self.action_head.reset_weights()
        self.critic_head.reset_weights()

    def forward(self, observations, input_actions=None):
        """ Calculate model outputs """
        observations = observations.float()
        value_hidden = self.value_backbone(observations)

        if input_actions is not None:
            action = input_actions
            value = self.critic_head(value_hidden, input_actions)
        else:
            policy_hidden = self.policy_backbone(observations)
            action = self.action_head(policy_hidden)
            value = self.critic_head(value_hidden, action)

        return action, value

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
            [self.value_backbone, self.critic_head]
        ]

    def step(self, observation):
        """ Select actions based on model's output """
        action, value = self(observation)

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
        observations = observations.float()
        policy_hidden = self.policy_backbone(observations)
        action = self.action_head(policy_hidden)
        return action

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return DeterministicPolicyEvaluator(self, rollout)


class DeterministicPolicyModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, policy_backbone: ModelFactory, value_backbone: ModelFactory,
                 critic_hidden_dim: int=64, critic_normalization='layer', critic_activation='relu'):
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_normalization = critic_normalization
        self.critic_activation = critic_activation

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return DeterministicPolicyModel(
            policy_backbone=policy_backbone,
            value_backbone=value_backbone,
            action_space=extra_args['action_space'],
            critic_hidden_dim=self.critic_hidden_dim,
            critic_normalization=self.critic_normalization,
            critic_activation=self.critic_activation
        )


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory,
           critic_hidden_dim: int=64, critic_normalization='layer', critic_activation='relu'):
    """ Vel creation function """
    return DeterministicPolicyModelFactory(
        policy_backbone=policy_backbone, value_backbone=value_backbone,
        critic_hidden_dim=critic_hidden_dim, critic_normalization=critic_normalization,
        critic_activation=critic_activation
    )
