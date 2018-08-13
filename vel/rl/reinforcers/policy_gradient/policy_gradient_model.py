import torch.nn.functional as F

import gym

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.exceptions import VelException
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead


class PolicyGradientModel(Model):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, backbone: LinearBackboneModel, action_space: gym.Space, argmax_sampling=False):
        super().__init__()
        self.argmax_sampling = argmax_sampling

        self.backbone = backbone
        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.backbone.output_dim,
            argmax_sampling=argmax_sampling
        )
        self.value_head = ValueHead(input_dim=self.backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        base_output = self.backbone(observations)

        action_output = self.action_head(base_output)
        value_output = self.value_head(base_output)

        return action_output, value_output

    def loss_value(self, x_data, y_true, y_pred):
        raise VelException("Invalid method to call for this model")

    def step(self, observation):
        """ Select actions based on model's output """
        action_logits, value_output = self(observation)
        actions = self.action_head.sample(action_logits)

        # - log probability
        neglogp = F.nll_loss(action_logits, actions, reduction='none')

        return actions, value_output, neglogp

    def value(self, observation):
        base_output = self.backbone(observation)
        value_output = self.value_head(base_output)
        return value_output

    def entropy(self, action_logits):
        return self.action_head.entropy(action_logits)


class PolicyGradientModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, backbone: ModelFactory, argmax_sampling=False):
        self.backbone = backbone
        self.argmax_sampling = argmax_sampling

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        backbone = self.backbone.instantiate(**extra_args)
        return PolicyGradientModel(backbone, extra_args['action_space'], self.argmax_sampling)


def create(backbone: ModelFactory, argmax_sampling=False):
    return PolicyGradientModelFactory(backbone=backbone, argmax_sampling=argmax_sampling)
