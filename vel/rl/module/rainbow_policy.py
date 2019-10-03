import gym
import torch

from vel.api import Network, BackboneNetwork
from vel.rl.module.head.q_distributional_noisy_dueling_head import QDistributionalNoisyDuelingHead


class RainbowPolicy(Network):
    """
    A deterministic greedy action-value model.
    Includes following commonly known modifications:
    - Distributional Q-Learning
    - Dueling architecture
    - Noisy Nets
    """

    def __init__(self, net: BackboneNetwork, action_space: gym.Space, vmin: float, vmax: float,
                 atoms: int = 1, initial_std_dev: float = 0.4, factorized_noise: bool = True):
        super().__init__()

        self.net = net

        self.action_space = action_space

        (value_size, adv_size) = self.net.size_hints().assert_tuple(2)

        self.q_head = QDistributionalNoisyDuelingHead(
            val_input_dim=value_size.last(),
            adv_input_dim=adv_size.last(),
            action_space=action_space,
            vmin=vmin, vmax=vmax, atoms=atoms,
            initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
        )

    @property
    def atom_delta(self) -> float:
        return self.q_head.atom_delta

    @property
    def support_atoms(self) -> torch.Tensor:
        return self.q_head.support_atoms

    def reset_weights(self):
        """ Initialize weights to reasonable defaults """
        self.net.reset_weights()
        self.q_head.reset_weights()

    def forward(self, observations):
        """ Model forward pass """
        advantage_features, value_features = self.net(observations)
        log_histogram = self.q_head(advantage_features, value_features)
        return log_histogram

    def histogram_info(self):
        """ Return extra information about histogram """
        return self.q_head.histogram_info()

    # def step(self, observations):
    #     """ Sample action from an action space for given state """
    #     log_histogram = self(observations)
    #     actions = self.q_head.sample(log_histogram)
    #
    #     return {
    #         'actions': actions,
    #         'log_histogram': log_histogram
    #     }

