import torch
import numpy as np
import torch.nn as nn

from vel.math.processes import OrnsteinUhlenbeckNoiseProcess
from vel.internals.generic_factory import GenericFactory


class OuNoise(nn.Module):
    """ Ornstein–Uhlenbeck noise process for action noise """

    def __init__(self, std_dev, environment):
        super().__init__()

        self.std_dev = std_dev
        self.action_space = environment.action_space
        self.processes = []

        self.register_buffer('low_tensor', torch.from_numpy(self.action_space.low).unsqueeze(0))
        self.register_buffer('high_tensor', torch.from_numpy(self.action_space.high).unsqueeze(0))

    def reset_training_state(self, dones, batch_info):
        """ A hook for a model to react when during training episode is finished """
        for idx, done in enumerate(dones):
            if done > 0.5:
                self.processes[idx].reset()

    def forward(self, actions, batch_info):
        """ Return model step after applying noise """
        while len(self.processes) < actions.shape[0]:
            len_action_space = self.action_space.shape[-1]

            self.processes.append(
                OrnsteinUhlenbeckNoiseProcess(
                    np.zeros(len_action_space), float(self.std_dev) * np.ones(len_action_space)
                )
            )

        noise = torch.from_numpy(np.stack([x() for x in self.processes])).float().to(actions.device)

        return torch.min(torch.max(actions + noise, self.low_tensor), self.high_tensor)


def create(std_dev: float):
    """ Vel factory function """
    return GenericFactory(OuNoise, arguments={'std_dev': std_dev})
