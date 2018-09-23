import torch
import torch.nn.functional as F

from vel.rl.api.base import OptimizerAlgoBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class DeepDeterministicPolicyGradient(OptimizerAlgoBase):
    """ Deep Deterministic Policy Gradient (DDPG) - policy gradient calculations """

    def __init__(self, model_factory, tau, max_grad_norm):
        super().__init__(max_grad_norm=max_grad_norm)

        self.model_factory = model_factory
        self.tau = tau

        self.discount_factor = None
        self.target_model = None

    def initialize(self, settings, model, environment, device):
        """ Initialize algo from reinforcer settings """
        self.discount_factor = settings.discount_factor

        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        with torch.no_grad():
            target_next_value = self.target_model.value(rollout['observations+1'])
            target_value = rollout['rewards'] + (1.0 - rollout['dones']) * self.discount_factor * target_next_value

        rollout_value = model.value(rollout['observations'], rollout['actions'])

        value_loss = F.mse_loss(rollout_value, target_value)
        policy_loss = -model.value(rollout['observations']).mean()

        loss = value_loss + policy_loss

        batch_info['sub_batch_data'].append({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'loss': loss.item()
        })

        return loss

    def post_optimization_step(self, batch_info, device, model, rollout):
        """ Steps to take after optimization has been done"""
        # Update target model
        for model_param, target_param in zip(model.parameters(), self.target_model.parameters()):
            # EWMA average model update
            target_param.data.mul_(1 - self.tau).add_(model_param.data * self.tau)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("value_loss"),
            AveragingNamedMetric("policy_loss"),
            AveragingNamedMetric("loss"),
        ]


def create(model, tau: float, max_grad_norm: float=None):
    return DeepDeterministicPolicyGradient(
        tau=tau,
        model_factory=model,
        max_grad_norm=max_grad_norm
    )
