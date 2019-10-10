import gym

from vel.api import VModule, BackboneModule

from vel.rl.module.head.stochastic_action_head import make_stockastic_action_head
from vel.rl.module.head.value_head import ValueHead


class StochasticPolicy(VModule):
    """
    Most generic policy gradient model class with a set of common actor-critic heads that share a single backbone
    """

    def __init__(self, net: BackboneModule, action_space: gym.Space):
        super().__init__()

        self.net = net

        assert not self.net.is_stateful, "Backbone shouldn't have state"

        (action_size, value_size) = self.net.size_hints().assert_tuple(2)

        self.action_head = make_stockastic_action_head(
            action_space=action_space,
            input_dim=action_size.last(),
        )

        self.value_head = ValueHead(
            input_dim=value_size.last()
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.action_head.reset_weights()
        self.value_head.reset_weights()

    def forward(self, observation):
        """ Calculate model outputs """
        action_hidden, value_hidden = self.net(observation)
        return self.action_head(action_hidden), self.value_head(value_hidden)
