import gym
import torch
import typing

from vel.api import LinearBackboneModel, ModelFactory, BackboneModel
from vel.module.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator, Policy
from vel.rl.module.stochastic_action_head import StochasticActionHead
from vel.rl.module.value_head import ValueHead


class StochasticPolicyEvaluator(Evaluator):
    """ Evaluator for a policy gradient model """

    def __init__(self, model: 'StochasticPolicy', rollout: Rollout):
        super().__init__(rollout)

        self.model = model

        pd_params, estimated_values = model(self.rollout.batch_tensor('observations'))

        self.provide('model:pd_params',  pd_params)
        self.provide('model:values', estimated_values)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout:actions')
        pd_params = self.get('model:pd_params')
        return self.model.action_head.logprob(actions, pd_params)

    @Evaluator.provides('model:entropy')
    def model_entropy(self):
        pd_params = self.get('model:pd_params')
        return self.model.action_head.entropy(pd_params)


class StochasticPolicy(Policy):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone

        assert not self.backbone.is_stateful, "Backbone shouldn't have state"

        self.action_head = StochasticActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )

        self.value_head = ValueHead(
            input_dim=self.backbone.output_dim
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observation):
        """ Calculate model outputs """
        input_data = self.input_block(observation)

        base_output = self.backbone(input_data)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.action_head.sample(action_pd_params, deterministic=deterministic)

        # log likelihood of selected action
        logprobs = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'action:logprobs': logprobs
        }

    def value(self, observation, state=None) -> torch.tensor:
        """ Calculate value only - small optimization """
        input_data = self.input_block(observation)
        base_output = self.backbone(input_data)
        return self.value_head(base_output)

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return StochasticPolicyEvaluator(self, rollout)


class StochasticPolicyFactory(ModelFactory):
    """ Factory class for policy gradient models """
    def __init__(self, input_block: IdentityFactory, backbone: ModelFactory):
        self.backbone = backbone
        self.input_block = input_block

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return StochasticPolicy(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory] = None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return StochasticPolicyFactory(input_block=input_block, backbone=backbone)
