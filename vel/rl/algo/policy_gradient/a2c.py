import torch
import torch.nn.functional as F

from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.math.functions import explained_variance
from vel.rl.api import OptimizerAlgoBase, Rollout, Trajectories
from vel.rl.discount_bootstrap import discount_bootstrap_gae


class A2CPolicyGradient(OptimizerAlgoBase):
    """ Simplest policy gradient - calculate loss as an advantage of an actor versus value function """
    def __init__(self, entropy_coefficient, value_coefficient, max_grad_norm, discount_factor: float, gae_lambda=1.0):
        super().__init__(max_grad_norm)

        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.gae_lambda = gae_lambda
        self.discount_factor = discount_factor

    def process_rollout(self, batch_info, rollout: Rollout):
        """ Process rollout for ALGO before any chunking/shuffling  """
        assert isinstance(rollout, Trajectories), "A2C requires trajectory rollouts"

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

        # Use evaluator interface to get the what we are interested in from the model
        advantages = evaluator.get('rollout:advantages')
        returns = evaluator.get('rollout:returns')
        rollout_values = evaluator.get('rollout:values')

        logprobs = evaluator.get('model:action:logprobs')
        values = evaluator.get('model:values')
        entropy = evaluator.get('model:entropy')

        # Actual calculations. Pretty trivial
        policy_loss = -torch.mean(advantages * logprobs)
        value_loss = 0.5 * F.mse_loss(values, returns)
        policy_entropy = torch.mean(entropy)

        loss_value = (
            policy_loss - self.entropy_coefficient * policy_entropy + self.value_coefficient * value_loss
        )

        loss_value.backward()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_entropy': policy_entropy.item(),
            'advantage_norm': torch.norm(advantages).item(),
            'explained_variance': explained_variance(returns, rollout_values)
        }

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_entropy"),
            AveragingNamedMetric("policy_loss"),
            AveragingNamedMetric("grad_norm"),
            AveragingNamedMetric("advantage_norm"),
            AveragingNamedMetric("explained_variance")
        ]


def create(entropy_coefficient, value_coefficient, max_grad_norm, discount_factor, gae_lambda=1.0):
    """ Vel factory function """
    return A2CPolicyGradient(
        entropy_coefficient,
        value_coefficient,
        max_grad_norm,
        discount_factor,
        gae_lambda
    )
