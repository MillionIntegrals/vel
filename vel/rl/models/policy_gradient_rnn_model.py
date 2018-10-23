import gym
import torch
import torch.nn as nn
import typing

from vel.api.base import RnnLinearBackboneModel, RnnModel, ModelFactory
from vel.rl.api import Rollout, Trajectories, Evaluator
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead


class PolicyGradientRnnEvaluator(Evaluator):
    """ Evaluate recurrent model from initial state """

    def __init__(self, model: 'PolicyGradientRnnModel', rollout: Rollout):
        assert isinstance(rollout, Trajectories), "For an RNN model, we must evaluate trajectories"
        super().__init__(rollout)

        self.model = model

        observation_trajectories = rollout.transition_tensors['observations']
        hidden_state = rollout.rollout_tensors['initial_hidden_state']

        action_accumulator = []
        value_accumulator = []

        # Evaluate recurrent network step by step
        for i in range(observation_trajectories.size(0)):
            action_output, value_output, hidden_state = model(observation_trajectories[i], hidden_state)

            action_accumulator.append(action_output)
            value_accumulator.append(value_output)

        policy_params = torch.cat(action_accumulator, dim=0)
        estimated_values = torch.cat(value_accumulator, dim=0)

        self.provide('model:policy_params', policy_params)
        self.provide('model:estimated_values', estimated_values)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout:actions')
        policy_params = self.get('model:policy_params')
        return self.model.action_head.logprob(actions, policy_params)

    @Evaluator.provides('model:entropy')
    def model_entropy(self):
        policy_params = self.get('model:policy_params')
        return self.model.entropy(policy_params)


class PolicyGradientRnnModel(RnnModel):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, backbone: RnnLinearBackboneModel, action_space: gym.Space,
                 input_block: typing.Optional[nn.Module]=None):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone

        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

        assert self.backbone.is_recurrent, "Backbone must be a recurrent model"

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        return self.backbone.state_dim

    def reset_weights(self):
        """ Initialize properly model weights """
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations, state):
        """ Calculate model outputs """
        if self.input_block is not None:
            input_data = self.input_block(observations)
        else:
            input_data = observations

        base_output, new_state = self.backbone(input_data, state=state)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output, new_state

    def step(self, observations, state, argmax_sampling=False):
        """ Select actions based on model's output """
        action_pd_params, value_output, new_state = self(observations, state)
        actions = self.action_head.sample(action_pd_params, argmax_sampling=argmax_sampling)

        # log likelihood of selected action
        logprobs = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'logprobs': logprobs,
            'state': new_state
        }

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return PolicyGradientRnnEvaluator(self, rollout)

    def logprob(self, action_sample, policy_params):
        """ Calculate - log(prob) of selected actions """
        return self.action_head.logprob(action_sample, policy_params)

    def value(self, observations, state):
        """ Calculate only value head for given state """
        if self.input_block is not None:
            input_data = self.input_block(observations)
        else:
            input_data = observations

        base_output, new_state = self.backbone(input_data, state)
        value_output = self.value_head(base_output)

        return value_output

    def entropy(self, action_pd_params):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(action_pd_params)


class PolicyGradientRnnModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, backbone: ModelFactory, input_block=None):
        self.backbone = backbone
        self.input_block = input_block

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        backbone = self.backbone.instantiate(**extra_args)

        if self.input_block is None:
            input_block = None
        else:
            input_block = self.input_block.instantiate()

        return PolicyGradientRnnModel(backbone, extra_args['action_space'], input_block)


def create(backbone: ModelFactory, input_block=None):
    """ Vel creation function """
    return PolicyGradientRnnModelFactory(backbone=backbone, input_block=input_block)
