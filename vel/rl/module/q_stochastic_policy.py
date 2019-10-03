import gym

from vel.api import BackboneNetwork, Network
from vel.rl.module.head.stochastic_action_head import make_stockastic_action_head
from vel.rl.module.head.q_head import QHead


class QStochasticPolicy(Network):
    """
    A policy model with an action-value critic head (instead of more common state-value critic head).
    Supports only discrete action spaces (ones that can be enumerated)
    """

    def __init__(self, net: BackboneNetwork, action_space: gym.Space):
        super().__init__()

        assert isinstance(action_space, gym.spaces.Discrete)

        self.net = net

        (action_size, value_size) = self.net.size_hints().assert_tuple(2)

        self.action_head = make_stockastic_action_head(
            input_dim=action_size.last(),
            action_space=action_space
        )

        self.q_head = QHead(
            input_dim=value_size.last(),
            action_space=action_space
        )

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.action_head.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Calculate model outputs """
        action_hidden, q_hidden = self.net(observations)
        policy_params = self.action_head(action_hidden)

        q = self.q_head(q_hidden)

        return policy_params, q
