import itertools as it

import gym

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.api import Rollout, Evaluator
from vel.rl.modules.action_head import ActionHead
from vel.rl.modules.value_head import ValueHead
from vel.rl.models.policy_gradient_model import PolicyGradientEvaluator


class PolicyGradientModelSeparate(Model):
    """ For a policy gradient algorithm we need set of custom heads for our model """

    def __init__(self, policy_backbone: LinearBackboneModel, value_backbone: LinearBackboneModel,
                 action_space: gym.Space):
        super().__init__()

        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = ActionHead(
            action_space=action_space,
            input_dim=self.policy_backbone.output_dim
        )

        self.value_head = ValueHead(input_dim=self.value_backbone.output_dim)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()

        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        policy_base_output = self.policy_backbone(observations)
        value_base_output = self.value_backbone(observations)

        action_output = self.action_head(policy_base_output)
        value_output = self.value_head(value_base_output)

        return action_output, value_output

    def step(self, observation, argmax_sampling=False):
        """ Select actions based on model's output """
        policy_params, values = self(observation)
        actions = self.action_head.sample(policy_params, argmax_sampling=argmax_sampling)

        # log likelihood of selected action
        logprobs = self.action_head.logprob(actions, policy_params)

        return {
            'actions': actions,
            'values': values,
            'logprobs': logprobs
        }

    def policy_parameters(self):
        """ Parameters of policy """
        return it.chain(self.policy_backbone.parameters(), self.action_head.parameters())

    def logprob(self, action_sample, policy_params):
        """ Calculate - log(prob) of selected actions """
        return self.action_head.logprob(action_sample, policy_params)

    def value(self, observation):
        """ Calculate only value head for given state """
        base_output = self.value_backbone(observation)
        value_output = self.value_head(base_output)
        return value_output

    def policy(self, observation):
        """ Calculate only action head for given state """
        policy_base_output = self.policy_backbone(observation)
        policy_params = self.action_head(policy_base_output)
        return policy_params

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        return PolicyGradientEvaluator(self, rollout)

    def entropy(self, policy_params):
        """ Entropy of a probability distribution """
        return self.action_head.entropy(policy_params)

    def kl_divergence(self, pd_q, pd_p):
        """ Calculate KL-divergence between two probability distributions """
        return self.action_head.kl_divergence(pd_q, pd_p)


class PolicyGradientModelSeparateFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, policy_backbone: ModelFactory, value_backbone: ModelFactory):
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return PolicyGradientModelSeparate(policy_backbone, value_backbone, extra_args['action_space'])


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory):
    return PolicyGradientModelSeparateFactory(
        policy_backbone=policy_backbone,
        value_backbone=value_backbone
    )
