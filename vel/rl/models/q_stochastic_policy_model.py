import gym
import torch
import typing

from vel.api import LinearBackboneModel, Model, ModelFactory, BackboneModel
from vel.modules.input.identity import IdentityFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.q_head import QHead


class QStochasticPolicyEvaluator(Evaluator):
    """ Evaluator for QPolicyGradientModel """
    def __init__(self, model: 'QStochasticPolicyModel', rollout: Rollout):
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


class QStochasticPolicyModel(Model):
    """
    A policy gradient model with an action-value critic head (instead of more common state-value critic head).
    Supports only discrete action spaces (ones that can be enumerated)
    """

    def __init__(self, input_block: BackboneModel, backbone: LinearBackboneModel, action_space: gym.Space):
        super().__init__()

        assert isinstance(action_space, gym.spaces.Discrete)

        self.input_block = input_block
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
        self.input_block.reset_weights()
        self.backbone.reset_weights()
        self.action_head.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        input_data = self.input_block(observations)

        base_output = self.backbone(input_data)
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
            'q': q,
            'logprobs': policy_params,
            'action:logprobs': logprobs
        }

    def evaluate(self, rollout: Rollout) -> QStochasticPolicyEvaluator:
        """ Evaluate model on a rollout """
        return QStochasticPolicyEvaluator(self, rollout)

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


class QStochasticPolicyModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, input_block: IdentityFactory, backbone: ModelFactory):
        self.backbone = backbone
        self.input_block = input_block

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        input_block = self.input_block.instantiate()
        backbone = self.backbone.instantiate(**extra_args)

        return QStochasticPolicyModel(input_block, backbone, extra_args['action_space'])


def create(backbone: ModelFactory, input_block: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    if input_block is None:
        input_block = IdentityFactory()

    return QStochasticPolicyModelFactory(input_block=input_block, backbone=backbone)
