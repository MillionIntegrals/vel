import torch
import torch.nn.utils

from vel.api import ModelFactory
from vel.api.metrics.averaging_metric import AveragingNamedMetric
from vel.rl.api import OptimizerAlgoBase


class DistributionalDeepQLearning(OptimizerAlgoBase):
    """ Deep Q-Learning algorithm """

    def __init__(self, model_factory: ModelFactory, discount_factor: float, double_dqn: bool,
                 target_update_frequency: int, max_grad_norm: float):
        super().__init__(max_grad_norm)

        self.model_factory = model_factory
        self.discount_factor = discount_factor

        self.double_dqn = double_dqn
        self.target_update_frequency = target_update_frequency

        self.target_model = None

        self.vmin = None
        self.vmax = None
        self.num_atoms = None
        self.support_atoms = None
        self.atom_delta = None

    def initialize(self, training_info, model, environment, device):
        """ Initialize policy gradient from reinforcer settings """
        self.target_model = self.model_factory.instantiate(action_space=environment.action_space).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        histogram_info = model.histogram_info()

        self.vmin = histogram_info['vmin']
        self.vmax = histogram_info['vmax']

        self.num_atoms = histogram_info['num_atoms']

        self.support_atoms = histogram_info['support_atoms']
        self.atom_delta = histogram_info['atom_delta']

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        evaluator = model.evaluate(rollout)
        batch_size = rollout.frames()

        dones_tensor = evaluator.get('rollout:dones')
        rewards_tensor = evaluator.get('rollout:rewards')

        assert dones_tensor.dtype == torch.float32

        with torch.no_grad():
            target_evaluator = self.target_model.evaluate(rollout)

            if self.double_dqn:
                # DOUBLE DQN
                # Histogram gets returned as logits initially, we need to exp it before projection
                target_value_histogram_for_all_actions = target_evaluator.get('model:q_dist_next').exp()
                model_value_histogram_for_all_actions = evaluator.get('model:q_dist_next').exp()

                atoms_aligned = self.support_atoms.view(1, 1, self.num_atoms)

                selected_action_indices = (
                    (atoms_aligned * model_value_histogram_for_all_actions).sum(dim=-1).argmax(dim=1)
                )

                # Select largest 'target' value based on action that 'model' selects
                next_value_histograms = (
                    target_value_histogram_for_all_actions[range(batch_size), selected_action_indices]
                )
            else:
                # REGULAR DQN
                # Histogram gets returned as logits initially, we need to exp it before projection
                target_value_histogram_for_all_actions = target_evaluator.get('model:q_dist_next').exp()

                atoms_aligned = self.support_atoms.view(1, 1, self.num_atoms)

                selected_action_indices = (
                    (atoms_aligned * target_value_histogram_for_all_actions).sum(dim=-1).argmax(dim=1)
                )

                next_value_histograms = (
                    target_value_histogram_for_all_actions[range(batch_size), selected_action_indices]
                )

            # HISTOGRAM PROJECTION CODE
            forward_steps = rollout.extra_data.get('forward_steps', 1)

            atoms_projected = (
                rewards_tensor.unsqueeze(1) +
                (self.discount_factor ** forward_steps) *
                (1 - dones_tensor).unsqueeze(1) * self.support_atoms.unsqueeze(0)
            )

            atoms_projected = atoms_projected.clamp(min=self.vmin, max=self.vmax)
            projection_indices = (atoms_projected - self.vmin) / self.atom_delta

            index_floor = projection_indices.floor().long()
            index_ceil = projection_indices.ceil().long()

            # Fix corner case when index_floor == index_ceil
            index_floor[(index_ceil > 0) * (index_floor == index_ceil)] -= 1
            index_ceil[(index_floor < (self.num_atoms - 1)) * (index_floor == index_ceil)] += 1

            value_histogram_projected = torch.zeros_like(next_value_histograms)

            # Following part will be a bit convoluted, in an effort to fully vectorize projection operation

            # Special offset index tensor
            offsets = (
                torch.arange(0, batch_size * self.num_atoms, self.num_atoms)
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .contiguous().view(-1).to(device)
            )

            # Linearize all the buffers
            value_histogram_projected = value_histogram_projected.view(-1)
            index_ceil = index_ceil.view(-1)
            index_floor = index_floor.view(-1)
            projection_indices = projection_indices.view(-1)

            value_histogram_projected.index_add_(
                0,
                index_floor+offsets,
                (next_value_histograms.view(-1) * (index_ceil.float() - projection_indices))
            )

            value_histogram_projected.index_add_(
                0,
                index_ceil+offsets,
                (next_value_histograms.view(-1) * (projection_indices - index_floor.float()))
            )

            value_histogram_projected = value_histogram_projected.reshape(next_value_histograms.shape)

        q_log_histogram_selected = evaluator.get('model:action:q_dist')

        # Cross-entropy loss as usual
        original_losses = -(value_histogram_projected * q_log_histogram_selected).sum(dim=1)

        if evaluator.is_provided('rollout:weights'):
            weights = evaluator.get('rollout:weights')
        else:
            weights = torch.ones_like(rewards_tensor)

        loss_value = torch.mean(weights * original_losses)
        loss_value.backward()

        with torch.no_grad():
            mean_q_model = (self.support_atoms.unsqueeze(0) * torch.exp(q_log_histogram_selected)).sum(dim=1).mean()
            mean_q_target = (self.support_atoms.unsqueeze(0) * value_histogram_projected).sum(dim=1).mean()

        return {
            'loss': loss_value.item(),
            # We need it to update priorities in the replay buffer:
            'errors': original_losses.detach().cpu().numpy(),
            'average_q_selected': mean_q_model.item(),
            'average_q_target': mean_q_target.item()
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
           max_grad_norm: float, double_dqn: bool = False):
    """ Vel factory function """
    return DistributionalDeepQLearning(
        model_factory=model,
        discount_factor=discount_factor,
        double_dqn=double_dqn,
        target_update_frequency=target_update_frequency,
        max_grad_norm=max_grad_norm
    )
