import torch
import torch.nn.functional as F
import torch.nn.utils

from vel.api import ModelFactory
from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.api import OptimizerAlgoBase


class DeepQLearning(OptimizerAlgoBase):
    """ Deep Q-Learning algorithm """

    def __init__(self, model_factory: ModelFactory, discount_factor: float, double_dqn: bool, target_update_frequency: int,
                 max_grad_norm: float):
        super().__init__(max_grad_norm)

        self.model_factory = model_factory
        self.discount_factor = discount_factor

        self.double_dqn = double_dqn
        self.target_update_frequency = target_update_frequency

        self.target_model = None

    def initialize(self, training_info, model, environment, device):
        """ Initialize policy gradient from reinforcer settings """
        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        evaluator = model.evaluate(rollout)

        dones_tensor = evaluator.get('rollout:dones')
        rewards_tensor = evaluator.get('rollout:rewards')

        assert dones_tensor.dtype == torch.float32

        with torch.no_grad():
            target_evaluator = self.target_model.evaluate(rollout)

            if self.double_dqn:
                # DOUBLE DQN
                target_q = target_evaluator.get('model:q_next')
                model_q = evaluator.get('model:q_next')
                # Select largest 'target' value based on action that 'model' selects
                values = target_q.gather(1, model_q.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                # REGULAR DQN
                # [0] is because in pytorch .max(...) returns tuple (max values, argmax)
                values = target_evaluator.get('model:q_next').max(dim=1)[0]

            forward_steps = rollout.extra_data.get('forward_steps', 1)
            estimated_return = rewards_tensor + (self.discount_factor ** forward_steps) * values * (1 - dones_tensor)

        q_selected = evaluator.get('model:action:q')

        if evaluator.is_provided('rollout:weights'):
            weights = evaluator.get('rollout:weights')
        else:
            weights = torch.ones_like(rewards_tensor)

        original_losses = F.smooth_l1_loss(q_selected, estimated_return, reduction='none')

        loss_value = torch.mean(weights * original_losses)
        loss_value.backward()

        return {
            'loss': loss_value.item(),
            # We need it to update priorities in the replay buffer:
            'errors': original_losses.detach().cpu().numpy(),
            'average_q_selected': torch.mean(q_selected).item(),
            'average_q_target': torch.mean(estimated_return).item()
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


def create(model: ModelFactory, discount_factor: float, target_update_frequency: int,
           max_grad_norm: float, double_dqn: bool=False):
    """ Vel factory function """
    return DeepQLearning(
        model_factory=model,
        discount_factor=discount_factor,
        double_dqn=double_dqn,
        target_update_frequency=target_update_frequency,
        max_grad_norm=max_grad_norm
    )
