import torch
import torch.nn.functional as F

from .policy_gradient_reinforcer import PolicyGradientBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class A2CPolicyGradient(PolicyGradientBase):
    """ Simplest policy gradient - calculate loss as an advantage of an actor versus value function """
    def __init__(self, entropy_coefficient, value_coefficient):
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient

    def calculate_loss(self, batch_info, device, model, rollout, data_dict=None):
        """ Calculate loss of the supplied rollout """
        observations = rollout['observations']
        discounted_rewards = rollout['discounted_rewards']
        advantages = rollout['advantages']
        actions = rollout['actions']

        action_logits, value_outputs = model(observations)

        neglogps = F.nll_loss(action_logits, actions, reduction='none')

        policy_gradient_loss = torch.mean(advantages * neglogps)
        value_loss = 0.5 * F.mse_loss(value_outputs, discounted_rewards)
        policy_entropy = torch.mean(model.entropy(action_logits))

        loss_value = (
            policy_gradient_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        data_dict['policy_loss'] = policy_gradient_loss
        data_dict['value_loss'] = value_loss
        data_dict['policy_entropy'] = policy_entropy

        return loss_value

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("policy_loss")
        ]


def create(entropy_coefficient, value_coefficient):
    return A2CPolicyGradient(entropy_coefficient, value_coefficient)
