"""
Code based on:
https://github.com/Kaixhin/Rainbow/blob/master/model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def scaled_noise(size, device):
    x = torch.randn(size, device=device)
    return x.sign().mul_(x.abs().sqrt_())


def factorized_gaussian_noise(in_features, out_features, device):
    """
    Factorised (cheaper) gaussian noise from "Noisy Networks for Exploration"
    by Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot and others
    """
    in_noise = scaled_noise(in_features, device=device)
    out_noise = scaled_noise(out_features, device=device)

    return out_noise.ger(in_noise), out_noise


def gaussian_noise(in_features, out_features, device):
    """ Normal gaussian N(0, 1) noise """
    return torch.randn((in_features, out_features), device=device), torch.randn(out_features, device=device)


class NoisyLinear(nn.Module):
    """ NoisyNets noisy linear layer """
    def __init__(self, in_features, out_features, initial_std_dev: float = 0.4, factorized_noise: bool = True):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

    def reset_weights(self):
        init.orthogonal_(self.weight_mu, gain=math.sqrt(2))
        init.constant_(self.bias_mu, 0.0)

        # Initialize the "random weights" to constants
        self.weight_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.initial_std_dev / math.sqrt(self.out_features))

    def forward(self, input_data):
        if self.training:
            if self.factorized_noise:
                weight_epsilon, bias_epsilon = factorized_gaussian_noise(
                    self.in_features, self.out_features, device=input_data.device
                )
            else:
                weight_epsilon, bias_epsilon = gaussian_noise(
                    self.in_features, self.out_features, device=input_data.device
                )

            return F.linear(
                input_data,
                self.weight_mu + self.weight_sigma * weight_epsilon,
                self.bias_mu + self.bias_sigma * bias_epsilon
            )
        else:
            return F.linear(input_data, self.weight_mu, self.bias_mu)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return (
            f'{self.in_features}, {self.out_features}, initial_std_dev={self.initial_std_dev}, '
            'factorized_noise={self.factorized_noise} '
        )
