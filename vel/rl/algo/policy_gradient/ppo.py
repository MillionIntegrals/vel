import torch

import numbers

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.api.base import OptimizerAlgoBase
from vel.math.functions import explained_variance
from vel.schedules.constant import ConstantSchedule


class PpoPolicyGradient(OptimizerAlgoBase):
    """ Proximal Policy Optimization - https://arxiv.org/abs/1707.06347 """
    def __init__(self, entropy_coefficient, value_coefficient, cliprange, max_grad_norm):
        super().__init__(max_grad_norm)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient

        if isinstance(cliprange, numbers.Number):
            self.cliprange = ConstantSchedule(cliprange)
        else:
            self.cliprange = cliprange

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        observations = rollout['observations']
        returns = rollout['returns']
        advantages = rollout['advantages']
        rollout_values = rollout['values']
        rollout_actions = rollout['actions']
        rollout_logprobs = rollout['logprobs']

        # Select the cliprange
        current_cliprange = self.cliprange.value(batch_info['progress'])

        # Normalize the advantages?
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PART 0 - model_evaluation
        eval_action_pd_params, eval_value_outputs = model(observations)

        # PART 1 - policy entropy
        policy_entropy = torch.mean(model.entropy(eval_action_pd_params))

        # PART 2 - value function
        value_output_clipped = rollout_values + torch.clamp(eval_value_outputs - rollout_values, -current_cliprange, current_cliprange)
        value_loss_part1 = (eval_value_outputs - returns).pow(2)
        value_loss_part2 = (value_output_clipped - returns).pow(2)
        value_loss = 0.5 * torch.mean(torch.max(value_loss_part1, value_loss_part2))

        # PART 3 - policy gradient loss
        eval_logprobs = model.logprob(rollout_actions, eval_action_pd_params)
        ratio = torch.exp(eval_logprobs - rollout_logprobs)

        pg_loss_part1 = -advantages * ratio
        pg_loss_part2 = -advantages * torch.clamp(ratio, 1.0 - current_cliprange, 1.0 + current_cliprange)
        policy_loss = torch.mean(torch.max(pg_loss_part1, pg_loss_part2))

        loss_value = (
                policy_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        loss_value.backward()

        with torch.no_grad():
            approx_kl_divergence = 0.5 * torch.mean((eval_logprobs - rollout_logprobs) ** 2)
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > current_cliprange).to(dtype=torch.float))

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'approx_kl_divergence': approx_kl_divergence.item(),
            'clip_fraction': clip_fraction.item(),
            'advantage_norm': torch.norm(advantages).item(),
            'explained_variance': explained_variance(returns, rollout_values)
        }

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("policy_loss"),
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("approx_kl_divergence"),
            AveragingNamedMetric("clip_fraction"),
            AveragingNamedMetric("grad_norm"),
            AveragingNamedMetric("advantage_norm"),
            AveragingNamedMetric("explained_variance")
        ]


def create(entropy_coefficient, value_coefficient, cliprange, max_grad_norm):
    return PpoPolicyGradient(entropy_coefficient, value_coefficient, cliprange, max_grad_norm)
