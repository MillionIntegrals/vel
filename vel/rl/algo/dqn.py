import torch
import torch.nn.functional as F
import torch.nn.utils

from vel.api.base import ModelFactory
from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.api.base import OptimizerAlgoBase


class DeepQLearning(OptimizerAlgoBase):
    """ Deep Q-Learning algorithm """

    def __init__(self, model_factory: ModelFactory, double_dqn: bool,
                 target_update_frequency: int, max_grad_norm: float):
        super().__init__(max_grad_norm)

        self.model_factory = model_factory

        self.double_dqn = double_dqn
        self.target_update_frequency = target_update_frequency

        self.discount_factor = None
        self.target_model = None

    def initialize(self, settings, model, environment, device):
        """ Initialize policy gradient from reinforcer settings """
        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.discount_factor = settings.discount_factor

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        observation_tensor = rollout['observations']
        observation_tensor_tplus1 = rollout['observations+1']
        dones_tensor = rollout['dones']
        rewards_tensor = rollout['rewards']
        actions_tensor = rollout['actions']

        with torch.no_grad():
            if self.double_dqn:
                # DOUBLE DQN
                target_values = self.target_model(observation_tensor_tplus1)
                model_values = model(observation_tensor_tplus1)
                # Select largest 'target' value based on action that 'model' selects
                values = target_values.gather(1, model_values.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                # REGULAR DQN
                values = self.target_model(observation_tensor_tplus1).max(dim=1)[0]

            expected_q = rewards_tensor + self.discount_factor * values * (1 - dones_tensor.float())

        q = model(observation_tensor)
        q_selected = q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        original_losses = F.smooth_l1_loss(q_selected, expected_q.detach(), reduction='none')

        weights = rollout['weights']

        loss_value = torch.mean(weights * original_losses)
        loss_value.backward()

        return {
            'loss': loss_value.item(),
            # We need it to update priorities in the replay buffer:
            'errors': original_losses.detach().cpu().numpy(),
            'average_q_selected': torch.mean(q_selected).item(),
            'average_q_target': torch.mean(expected_q).item()
        }

    def post_optimization_step(self, batch_info, device, model, rollout):
        """ Steps to take after optimization has been done"""
        if batch_info.aggregate_batch_number % self.target_update_frequency == 0:
            self.target_model.load_state_dict(model.state_dict())
            self.target_model.eval()

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("loss"),
            AveragingNamedMetric("average_q_selected"),
            AveragingNamedMetric("average_q_target"),
            AveragingNamedMetric("grad_norm"),
        ]


def create(model: ModelFactory, target_update_frequency: int,
           max_grad_norm: float, double_dqn: bool=False):
    return DeepQLearning(
        model_factory=model,
        double_dqn=double_dqn,
        target_update_frequency=target_update_frequency,
        max_grad_norm=max_grad_norm
    )
