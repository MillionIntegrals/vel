import torch
import torch.autograd
import torch.nn.functional as F

from vel.rl.api.base import AlgoBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class DeepDeterministicPolicyGradient(AlgoBase):
    """ Deep Deterministic Policy Gradient (DDPG) - policy gradient calculations """

    def __init__(self, model_factory, tau, max_grad_norm):
        super().__init__()

        self.model_factory = model_factory
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.discount_factor = None
        self.target_model = None

    def initialize(self, settings, model, environment, device):
        """ Initialize algo from reinforcer settings """
        self.discount_factor = settings.discount_factor

        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        # Calculate value loss - or critic loss
        with torch.no_grad():
            target_next_value = self.target_model.value(rollout['observations+1'])
            target_value = rollout['rewards'] + (1.0 - rollout['dones']) * self.discount_factor * target_next_value

        # It may seem a bit tricky what I'm doing here, but the underlying idea is simple
        # All other implementations I found keep two separate optimizers for actor and critic
        # and update them separately
        # What I'm trying to do is to optimize them both with a single optimizer
        # but I need to make sure gradients flow correctly
        # From critic loss to critic network only and from actor loss to actor network only

        rollout_value = model.value(rollout['observations'], rollout['actions'])
        value_loss = F.mse_loss(rollout_value, target_value)

        # Backpropagate value loss to critic only
        value_loss.backward()

        model_action = model.action(rollout['observations'])
        model_action_value = model.value(rollout['observations'], model_action)

        policy_loss = -model_action_value.mean()

        model_action_grad = torch.autograd.grad(policy_loss, model_action)[0]

        # Backpropagate actor loss to actor only
        model_action.backward(gradient=model_action_grad)

        batch_info['sub_batch_data'].append({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        })

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        batch_info.optimizer.zero_grad()

        self.calculate_loss(batch_info=batch_info, device=device, model=model, rollout=rollout)

        self._clip_gradients(batch_info, model, self.max_grad_norm)

        batch_info.optimizer.step(closure=None)

        self.post_optimization_step(batch_info, device, model, rollout)

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
        ]


def create(model, tau: float, max_grad_norm: float=None):
    return DeepDeterministicPolicyGradient(
        tau=tau,
        model_factory=model,
        max_grad_norm=max_grad_norm
    )
