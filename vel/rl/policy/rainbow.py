import gym
import torch
import torch.nn.utils

from vel.api import ModuleFactory, BackboneModule, BatchInfo, OptimizerFactory, VelOptimizer
from vel.metric import AveragingNamedMetric
from vel.rl.api import RlPolicy, Rollout
from vel.rl.module.rainbow_policy import RainbowPolicy
from vel.util.situational import gym_space_to_size_hint


class Rainbow(RlPolicy):
    """ Deep Q-Learning algorithm """

    def __init__(self, net: BackboneModule, target_net: BackboneModule, action_space: gym.Space,
                 discount_factor: float, target_update_frequency: int, vmin: float, vmax: float, atoms: int = 1,
                 initial_std_dev: float = 0.4, factorized_noise: bool = True):
        super().__init__(discount_factor)

        self.net = RainbowPolicy(
            net=net,
            action_space=action_space,
            vmin=vmin,
            vmax=vmax,
            atoms=atoms,
            initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise
        )

        self.target_net = RainbowPolicy(
            net=target_net,
            action_space=action_space,
            vmin=vmin,
            vmax=vmax,
            atoms=atoms,
            initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise
        )
        self.target_net.requires_grad_(False)

        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency

        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = atoms

        # self.support_atoms = self.model.q
        # self.atom_delta = histogram_info['atom_delta']
        self.register_buffer('support_atoms', self.net.support_atoms.clone())
        self.atom_delta = self.net.atom_delta

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        return optimizer_factory.instantiate(parameters)

    def train(self, mode=True):
        """ Override train to make sure target model is always in eval mode """
        self.net.train(mode)
        self.target_net.train(False)

    def reset_weights(self):
        """ Initialize properly model weights """
        self.net.reset_weights()
        self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, observation, state=None):
        """ Calculate model outputs """
        return self.net(observation)

    def act(self, observation, state=None, deterministic=False):
        """ Select actions based on model's output """
        self.train(mode=not deterministic)

        q_values = self.net(observation)
        actions = self.net.q_head.sample(q_values)

        return {
            'actions': actions,
            'q': q_values
        }

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """ Calculate loss of the supplied rollout """
        batch_size = rollout.frames()

        observations = rollout.batch_tensor('observations')
        observations_next = rollout.batch_tensor('observations_next')

        actions = rollout.batch_tensor('actions')
        dones_tensor = rollout.batch_tensor('dones')
        rewards_tensor = rollout.batch_tensor('rewards')

        assert dones_tensor.dtype == torch.float32

        q = self.net(observations)

        with torch.no_grad():
            # DOUBLE DQN
            # Histogram gets returned as logits initially, we need to exp it before projection
            target_value_histogram_for_all_actions = self.target_net(observations_next).exp()
            model_value_histogram_for_all_actions = self.net(observations_next).exp()

            atoms_aligned = self.support_atoms.view(1, 1, self.num_atoms)

            selected_action_indices = (
                (atoms_aligned * model_value_histogram_for_all_actions).sum(dim=-1).argmax(dim=1)
            )

            # Select largest 'target' value based on action that 'model' selects
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
                .contiguous().view(-1).to(value_histogram_projected.device)
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

        q_log_histogram_selected = q[range(q.size(0)), actions]

        # Cross-entropy loss as usual
        original_losses = -(value_histogram_projected * q_log_histogram_selected).sum(dim=1)

        if rollout.has_tensor('weights'):
            weights = rollout.batch_tensor('weights')
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

    def post_optimization_step(self, batch_info, rollout):
        """ Steps to take after optimization has been done"""
        if batch_info.aggregate_batch_number % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return [
            AveragingNamedMetric("loss"),
            AveragingNamedMetric("average_q_selected"),
            AveragingNamedMetric("average_q_target")
        ]


class RainbowFactory(ModuleFactory):
    def __init__(self, net_factory: ModuleFactory, discount_factor: float, target_update_frequency: int,
                 vmin: float, vmax: float, atoms: int = 1, initial_std_dev: float = 0.4, factorized_noise: bool = True):
        self.net_factory = net_factory
        self.discount_factor = discount_factor
        self.target_update_frequency = target_update_frequency
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = atoms
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise

    def instantiate(self, **extra_args):
        """ Instantiate the model """
        action_space = extra_args.pop('action_space')
        observation_space = extra_args.pop('observation_space')

        size_hint = gym_space_to_size_hint(observation_space)

        # TODO(jerry): Push noisy net parameters down the stack here
        net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)
        target_net = self.net_factory.instantiate(size_hint=size_hint, **extra_args)

        return Rainbow(
            net=net,
            target_net=target_net,
            action_space=action_space,
            discount_factor=self.discount_factor,
            target_update_frequency=self.target_update_frequency,
            vmin=self.vmin,
            vmax=self.vmax,
            atoms=self.atoms,
            initial_std_dev=self.initial_std_dev,
            factorized_noise=self.factorized_noise
        )


def create(net: ModuleFactory, discount_factor: float, target_update_frequency: int,
           vmin: float, vmax: float, atoms: int = 1, initial_std_dev: float = 0.4, factorized_noise: bool = True):
    """ Vel factory function """
    return RainbowFactory(
        net_factory=net,
        discount_factor=discount_factor,
        target_update_frequency=target_update_frequency,
        vmin=vmin,
        vmax=vmax,
        atoms=atoms,
        initial_std_dev=initial_std_dev,
        factorized_noise=factorized_noise
    )
