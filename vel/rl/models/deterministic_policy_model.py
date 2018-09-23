import gym
import itertools as it

from vel.api.base import LinearBackboneModel, Model, ModelFactory
from vel.rl.modules.deterministic_action_head import DeterministicActionHead
from vel.rl.modules.deterministic_critic_head import DeterministicCriticHead


class DeterministicPolicyModel(Model):
    """ Deterministic Policy Gradient - model """

    def __init__(self, policy_backbone: LinearBackboneModel, value_backbone: LinearBackboneModel,
                 action_space: gym.Space):
        super().__init__()

        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

        self.action_head = DeterministicActionHead(self.policy_backbone.output_dim, action_space)
        self.critic_head = DeterministicCriticHead(self.value_backbone.output_dim, action_space)

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


class DeterministicPolicyModelFactory(ModelFactory):
    """ Factory  class for policy gradient models """
    def __init__(self, policy_backbone: ModelFactory, value_backbone: ModelFactory):
        self.policy_backbone = policy_backbone
        self.value_backbone = value_backbone

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        policy_backbone = self.policy_backbone.instantiate(**extra_args)
        value_backbone = self.value_backbone.instantiate(**extra_args)

        return DeterministicPolicyModel(
            policy_backbone=policy_backbone,
            value_backbone=value_backbone,
            action_space=extra_args['action_space']
        )


def create(policy_backbone: ModelFactory, value_backbone: ModelFactory):
    """ Vel creation function """
    return DeterministicPolicyModelFactory(policy_backbone=policy_backbone, value_backbone=value_backbone)
