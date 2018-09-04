import torch
import torch.nn.functional as F

from vel.rl.reinforcers.policy_gradient.policy_gradient_base import OptimizerPolicyGradientBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class A2CPolicyGradient(OptimizerPolicyGradientBase):
    """ Simplest policy gradient - calculate loss as an advantage of an actor versus value function """
    def __init__(self, entropy_coefficient, value_coefficient, max_grad_norm):
        super().__init__(max_grad_norm)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient

    def calculate_loss(self, batch_info, device, model, rollout):
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

        batch_info['policy_gradient_data'].append({
            'policy_loss': policy_gradient_loss,
            'value_loss': value_loss,
            'policy_entropy': policy_entropy
        })

        return loss_value

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("policy_loss")
        ]


def create(entropy_coefficient, value_coefficient, max_grad_norm):
    return A2CPolicyGradient(entropy_coefficient, value_coefficient, max_grad_norm)
