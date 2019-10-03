import gym

from vel.api import Network, BackboneNetwork

from vel.rl.module.head.q_head import QHead
from vel.rl.module.head.q_dueling_head import QDuelingHead


class QPolicy(Network):
    """
    Simple deterministic greedy action-value model.
    Supports only discrete action spaces (ones that can be enumerated)
    """
    def __init__(self, net: BackboneNetwork, action_space: gym.Space, dueling_dqn=False):
        super().__init__()

        self.dueling_dqn = dueling_dqn
        self.action_space = action_space

        self.net = net

        if self.dueling_dqn:
            (value_size, adv_size) = self.net.size_hints().assert_tuple(2)

            self.q_head = QDuelingHead(
                val_input_dim=value_size.last(),
                adv_input_dim=adv_size.last(),
                action_space=action_space
            )
        else:
            self.q_head = QHead(
                input_dim=self.net.size_hints().assert_single(2).last(),
                action_space=action_space
            )

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.net.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Model forward pass """
        if self.dueling_dqn:
            val_output, adv_output = self.net(observations)
            q_values = self.q_head(val_output, adv_output)
        else:
            base_output = self.net(observations)
            q_values = self.q_head(base_output)

        return q_values
