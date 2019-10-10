import itertools as it

import gym

from vel.api import VModule, BackboneModule


from vel.rl.module.head.deterministic_action_head import DeterministicActionHead
from vel.rl.module.head.deterministic_critic_head import DeterministicCriticHead


class ActorCriticPolicy(VModule):
    """ Deterministic Policy Gradient - model """

    def __init__(self, input_net: BackboneModule, policy_net: BackboneModule,
                 value_net: BackboneModule, action_space: gym.Space):
        super().__init__()

        self.input_net = input_net
        self.policy_backbone = policy_net
        self.value_backbone = value_net

        self.action_head = DeterministicActionHead(
            input_dim=self.policy_backbone.size_hints().assert_single().last(),
            action_space=action_space
        )

        self.critic_head = DeterministicCriticHead(
            input_dim=self.value_backbone.size_hints().assert_single().last()
        )

    def layer_groups(self):
        """ Grouped layers for optimization purposes """
        return [
            [self.input_net, self.policy_backbone, self.action_head],
            [self.input_net, self.value_backbone, self.critic_head],
        ]

    def reset_weights(self):
        """ Initialize properly model weights """
        self.input_net.reset_weights()
        self.policy_backbone.reset_weights()
        self.value_backbone.reset_weights()
        self.action_head.reset_weights()
        self.critic_head.reset_weights()

    def forward(self, observations, input_actions=None):
        """ Calculate model outputs """
        observations = self.input_net(observations)

        if input_actions is not None:
            actions = input_actions

            value_hidden = self.value_backbone((observations, actions))

            values = self.critic_head(value_hidden)
        else:
            policy_hidden = self.policy_backbone(observations)
            actions = self.action_head(policy_hidden)

            # value_input = torch.cat([observations, actions], dim=1)
            value_hidden = self.value_backbone((observations, actions))

            values = self.critic_head(value_hidden)

        return actions, values

    def policy_parameters(self):
        """ Parameters of policy """
        return it.chain(self.input_net(), self.policy_backbone.parameters(), self.action_head.parameters())

    def value_parameters(self):
        """ Parameters of policy """
        return it.chain(self.input_net(), self.value_backbone.parameters(), self.critic_head.parameters())

    def value(self, observation, input_actions=None):
        """ Calculate value for given state """
        action, value = self(observation, input_actions)
        return value

    def action(self, observations):
        """ Calculate value for given state """
        observations = self.input_net(observations)
        policy_hidden = self.policy_backbone(observations)
        action = self.action_head(policy_hidden)
        return action
