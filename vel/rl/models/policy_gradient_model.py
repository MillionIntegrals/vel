import gym
import torch.nn as nn
import typing

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead


class PolicyGradientModel(Model):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, backbone: LinearBackboneModel, action_space: gym.Space,
                 input_block: typing.Optional[nn.Module]=None):
        super().__init__()

        self.input_block = input_block
        self.backbone = backbone
        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        if self.input_block is not None:
            input_data = self.input_block(observations)
        else:
            input_data = observations

        base_output = self.backbone(input_data)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        action_pd_params, value_output = self(observation)
        actions = self.action_head.sample(action_pd_params, argmax_sampling=argmax_sampling)

        # log likelihood of selected action
        logprob = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': value_output,
            'logprob': logprob
        }

    def logprob(self, action_sample, action_params):
        """ Calculate - log(prob) of selected actions """
        return self.action_head.logprob(action_sample, action_params)

    def value(self, observations):
        """ Calculate only value head for given state """
        if self.input_block is not None:
            input_data = self.input_block(observations)
        else:
            input_data = observations

        base_output = self.backbone(input_data)
        value_output = self.value_head(base_output)
        return value_output

    def entropy(self, action_pd_params):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(action_pd_params)


class PolicyGradientModelFactory(ModelFactory):
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

        return PolicyGradientModel(backbone, extra_args['action_space'], input_block)


def create(backbone: ModelFactory, input_block=None):
    """ Vel creation function """
    return PolicyGradientModelFactory(backbone=backbone, input_block=input_block)
