import gym
import torch

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.q_head import QHead


class QPolicyGradientEvaluator(Evaluator):
    """ Evaluator for QPolicyGradientModel """
    def __init__(self, model: 'QPolicyGradientModel', rollout: Rollout):
        super().__init__(rollout)

        self.model = model

        observations = self.get('rollout:observations')
        logprobs, q = model(observations)

        self.provide('model:logprobs', logprobs)
        self.provide('model:q', q)

    @Evaluator.provides('model:action:logprobs')
    def model_action_logprobs(self):
        actions = self.get('rollout_actions')
        logprobs = self.get('model:logprobs')
        return self.model.action_head.logprob(actions, logprobs)


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

        policy_params = self.action_head(base_output)
        q = self.q_head(base_output)

        return policy_params, q

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        policy_params, q = self(observation)
        actions = self.action_head.sample(policy_params, argmax_sampling=argmax_sampling)

        # log probability - we can do that, because we support only discrete action spaces
        logprobs = self.action_head.logprob(actions, policy_params)

        return {
            'actions': actions,
            'values': q,
            'logprobs': policy_params,
            'action_logprobs': logprobs
        }

    def evaluate(self, rollout: Rollout) -> QPolicyGradientEvaluator:
        """ Evaluate model on a rollout """
        return QPolicyGradientEvaluator(self, rollout)

        # observations = rollout.batch_tensor('observations')
        # actions = rollout.batch_tensor('actions')
        #
        # policy_params, q = self(observations)
        # logprobs = self.action_head.logprob(actions, policy_params)
        #
        # return {
        #     'action_pd_params': policy_params,
        #     'actiologprobs': logprobs,
        #     'q': q
        # }

    def value(self, observation):
        """ Calculate only value head for given state """
        policy_params, q = self(observation)

        # Expectation of Q value with respect to action
        return (torch.exp(policy_params) * q).sum(dim=1)

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
