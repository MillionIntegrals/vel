import gym
import torch
import torch.nn.functional as F

from vel.api.base import LinearBackboneModel, Model, ModelFactory

from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.q_head import QHead


class QPolicyGradientModel(Model):
    """ Custom heads for a policy gradient model with a replay buffer """

    def __init__(self, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        self.backbone = backbone

        self.action_head = ActionHead(
            input_dim=self.backbone.output_dim,
            action_space=action_space
        )

        self.q_head = QHead(
            input_dim=self.backbone.output_dim,
            action_space=action_space
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        base_output = self.backbone(observations)

        action_output = self.action_head(base_output)
        q_output = self.q_head(base_output)

        return action_output, q_output

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        action_logits, q_output = self(observation)
        actions = self.action_head.sample(action_logits, argmax_sampling=argmax_sampling)

        # - log probability
        neglogp = F.nll_loss(action_logits, actions, reduction='none')

        return {
            'actions': actions,
            'q': q_output,
            'action_logits': action_logits,
            'neglogp': neglogp
        }

    def value(self, observation):
        """ Calculate only value head for given state """
        final_action_logits, final_q = self(observation)

        # Expectation of Q value with respect to action
        return (torch.exp(final_action_logits) * final_q).sum(dim=1)

    def entropy(self, action_logits):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(action_logits)


class QPolicyGradientModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, backbone: ModelFactory):
        self.backbone = backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        backbone = self.backbone.instantiate(**extra_args)
        return QPolicyGradientModel(backbone, extra_args['action_space'])


def create(backbone: ModelFactory):
    return QPolicyGradientModelFactory(backbone=backbone)
