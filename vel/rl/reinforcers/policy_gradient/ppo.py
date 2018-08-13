import torch
import torch.nn.functional as F

from .policy_gradient_reinforcer import PolicyGradientBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class PpoPolicyGradient(PolicyGradientBase):
    """ Proximal Policy Optimization - https://arxiv.org/abs/1707.06347 """
    def __init__(self, entropy_coefficient, value_coefficient, cliprange, cliprange_scaling):
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient

        self.cliprange = cliprange  # This needs to be verified
        self.cliprange_scaling = cliprange_scaling

    def cliprange_scaling_function(self, batch_info):
        """ Select current cliprange"""

        if self.cliprange_scaling == 'constant':
            return self.cliprange
        elif self.cliprange_scaling == 'linear':
            return (1.0 - batch_info['progress']) * self.cliprange
        else:
            raise NotImplementedError

    def calculate_loss(self, batch_info, device, model, rollout, data_dict=None):
        """ Calculate loss of the supplied rollout """

        observations = rollout['observations']
        discounted_rewards = rollout['discounted_rewards']
        advantages = rollout['advantages']
        values = rollout['values']
        rollout_actions = rollout['actions']
        rollout_neglogps = rollout['neglogps']

        # Select the cliprange
        current_cliprange = self.cliprange_scaling_function(batch_info)

        # Normalize the advantages?
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PART 0 - model_evaluation
        eval_action_logits, eval_value_outputs = model(observations)

        # PART 1 - policy entropy
        policy_entropy = torch.mean(model.entropy(eval_action_logits))

        # PART 2 - value function
        value_output_clipped = values + torch.clamp(eval_value_outputs - values, -current_cliprange, current_cliprange)
        value_loss_part1 = (eval_value_outputs - discounted_rewards).pow(2)
        value_loss_part2 = (value_output_clipped - discounted_rewards).pow(2)
        value_loss = 0.5 * torch.mean(torch.max(value_loss_part1, value_loss_part2))

        # PART 3 - policy gradient loss
        eval_neglogps = F.nll_loss(eval_action_logits, rollout_actions, reduction='none')
        ratio = torch.exp(rollout_neglogps - eval_neglogps)

        pg_loss_part1 = -advantages * ratio
        pg_loss_part2 = -advantages * torch.clamp(ratio, 1.0 - current_cliprange, 1.0 + current_cliprange)
        policy_gradient_loss = torch.mean(torch.max(pg_loss_part1, pg_loss_part2))

        loss_value = (
                policy_gradient_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        with torch.no_grad():
            approx_kl_divergence = 0.5 * torch.mean((eval_neglogps - rollout_neglogps))
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > current_cliprange).to(dtype=torch.float))

        data_dict['policy_loss'] = policy_gradient_loss
        data_dict['value_loss'] = value_loss
        data_dict['policy_entropy'] = policy_entropy
        data_dict['approx_kl_divergence'] = approx_kl_divergence
        data_dict['clip_fraction'] = clip_fraction

        return loss_value

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("policy_loss"),
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("approx_kl_divergence"),
            AveragingNamedMetric("clip_fraction")
        ]


def create(entropy_coefficient, value_coefficient, cliprange, cliprange_scaling):
    return PpoPolicyGradient(entropy_coefficient, value_coefficient, cliprange, cliprange_scaling)
