import torch

import numbers

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.math.functions import explained_variance
from vel.rl.api import OptimizerAlgoBase, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae
from vel.schedules.constant import ConstantSchedule


class PpoPolicyGradient(OptimizerAlgoBase):
    """ Proximal Policy Optimization - https://arxiv.org/abs/1707.06347 """
    def __init__(self, entropy_coefficient, value_coefficient, cliprange, max_grad_norm, discount_factor: float,
                 normalize_advantage: bool=True, gae_lambda: float=1.0):
        super().__init__(max_grad_norm)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.normalize_advantage = normalize_advantage

        if isinstance(cliprange, numbers.Number):
            self.cliprange = ConstantSchedule(cliprange)
        else:
            self.cliprange = cliprange

        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor

    def process_rollout(self, batch_info, rollout: Rollout):
        """ Process rollout for ALGO before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "PPO requires trajectory rollouts"

        advantages = discount_bootstrap_gae(
            rewards_buffer=rollout.transition_tensors['rewards'],
            dones_buffer=rollout.transition_tensors['dones'],
            values_buffer=rollout.transition_tensors['values'],
            final_values=rollout.rollout_tensors['final_values'],
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
            number_of_steps=rollout.num_steps
        )

        returns = advantages + rollout.transition_tensors['values']

        rollout.transition_tensors['advantages'] = advantages
        rollout.transition_tensors['returns'] = returns

        return rollout

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        evaluator = model.evaluate(rollout)

        # Part 0.0 - Rollout values
        advantages = evaluator.get('rollout:advantages')
        rollout_values = evaluator.get('rollout:values')
        rollout_action_logprobs = evaluator.get('rollout:action:logprobs')
        returns = evaluator.get('rollout:returns')

        # PART 0.1 - Model evaluation
        entropy = evaluator.get('model:entropy')
        model_values = evaluator.get('model:values')
        model_action_logprobs = evaluator.get('model:action:logprobs')

        # Select the cliprange
        current_cliprange = self.cliprange.value(batch_info['progress'])

        # Normalize the advantages?
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PART 1 - policy entropy
        policy_entropy = torch.mean(entropy)

        # PART 2 - value function
        value_output_clipped = rollout_values + torch.clamp(
            model_values - rollout_values, -current_cliprange, current_cliprange
        )
        value_loss_part1 = (model_values - returns).pow(2)
        value_loss_part2 = (value_output_clipped - returns).pow(2)
        value_loss = 0.5 * torch.mean(torch.max(value_loss_part1, value_loss_part2))

        # PART 3 - policy gradient loss
        ratio = torch.exp(model_action_logprobs - rollout_action_logprobs)

        pg_loss_part1 = -advantages * ratio
        pg_loss_part2 = -advantages * torch.clamp(ratio, 1.0 - current_cliprange, 1.0 + current_cliprange)
        policy_loss = torch.mean(torch.max(pg_loss_part1, pg_loss_part2))

        loss_value = (
            policy_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        loss_value.backward()

        with torch.no_grad():
            approx_kl_divergence = 0.5 * torch.mean((model_action_logprobs - rollout_action_logprobs).pow(2))
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


def create(entropy_coefficient, value_coefficient, cliprange, max_grad_norm, discount_factor,
           normalize_advantage=True, gae_lambda=1.0):
    """ Vel factory function """
    return PpoPolicyGradient(
        entropy_coefficient, value_coefficient, cliprange, max_grad_norm,
        discount_factor=discount_factor,
        normalize_advantage=normalize_advantage,
        gae_lambda=gae_lambda
    )
