import gym
import torch

from vel.api.base import LinearBackboneModel, Model, ModelFactory

from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.q_head import QHead


class QPolicyGradientModel(Model):
    """ Custom heads for a policy gradient model with a action-value head """

    def __init__(self, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        assert isinstance(action_space, gym.spaces.Discrete)

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
        action_pd_params, q_output = self(observation)
        actions = self.action_head.sample(action_pd_params, argmax_sampling=argmax_sampling)

        # log probability - we can do that, because we support only discrete action spaces
        logprob = self.action_head.logprob(actions, action_pd_params)

        return {
            'actions': actions,
            'values': q_output,
            'action_logits': action_pd_params,
            'logprob': logprob
        }

    def value(self, observation):
        """ Calculate only value head for given state """
        action_pd_params, final_q = self(observation)

        # Expectation of Q value with respect to action
        return (torch.exp(action_pd_params) * final_q).sum(dim=1)

    def entropy(self, action_logits):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(action_logits)

    def kl_divergence(self, logits_q, logits_p):
        """ Calculate KL-divergence between two probability distributions """
        return self.action_head.kl_divergence(logits_q, logits_p)


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
