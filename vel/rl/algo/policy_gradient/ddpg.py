import torch
import typing
import torch.autograd
import torch.nn.functional as F

from vel.rl.api import OptimizerAlgoBase
from vel.api.metrics.averaging_metric import AveragingNamedMetric


class DeepDeterministicPolicyGradient(OptimizerAlgoBase):
    """ Deep Deterministic Policy Gradient (DDPG) - policy gradient calculations """

    def __init__(self, model_factory, discount_factor: float, tau: float, max_grad_norm: typing.Optional[float]=None):
        super().__init__(max_grad_norm)

        self.model_factory = model_factory
        self.tau = tau
        self.discount_factor = discount_factor

        self.target_model = None

    def initialize(self, training_info, model, environment, device):
        """ Initialize algo from reinforcer settings """
        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        rollout = rollout.to_transitions()

        dones = rollout.batch_tensor('dones')
        rewards = rollout.batch_tensor('rewards')
        observations_next = rollout.batch_tensor('observations_next')
        actions = rollout.batch_tensor('actions')
        observations = rollout.batch_tensor('observations')

        # Calculate value loss - or critic loss
        with torch.no_grad():
            target_next_value = self.target_model.value(observations_next)
            target_value = rewards + (1.0 - dones) * self.discount_factor * target_next_value

        # Value estimation error vs the target network
        model_value = model.value(observations, actions)
        value_loss = F.mse_loss(model_value, target_value)

        # It may seem a bit tricky what I'm doing here, but the underlying idea is simple
        # All other implementations I found keep two separate optimizers for actor and critic
        # and update them separately
        # What I'm trying to do is to optimize them both with a single optimizer
        # but I need to make sure gradients flow correctly
        # From critic loss to critic network only and from actor loss to actor network only

        # Backpropagate value loss to critic only
        value_loss.backward()

        model_action = model.action(observations)
        model_action_value = model.value(observations, model_action)

        policy_loss = -model_action_value.mean()

        model_action_grad = torch.autograd.grad(policy_loss, model_action)[0]

        # Backpropagate actor loss to actor only
        model_action.backward(gradient=model_action_grad)

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

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


def create(model, discount_factor: float, tau: float, max_grad_norm: float=None):
    """ Vel factory function """
    return DeepDeterministicPolicyGradient(
        tau=tau,
        discount_factor=discount_factor,
        model_factory=model,
        max_grad_norm=max_grad_norm
    )
