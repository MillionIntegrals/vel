import torch
import numpy as np

from vel.api import BatchInfo
from vel.rl.api import (
    Trajectories, Rollout, ReplayEnvRollerBase, ReplayEnvRollerFactoryBase, ReplayBuffer, ReplayBufferFactory, RlModel
)
from vel.util.tensor_accumulator import TensorAccumulator


class TrajectoryReplayEnvRoller(ReplayEnvRollerBase):
    """
    Calculate environment rollouts using a replay buffer for experience replay.
    Replay buffer is parametrized.
    Samples trajectories from the replay buffer (consecutive series of frames)
    """

    def __init__(self, environment, device, replay_buffer: ReplayBuffer):
        self._environment = environment
        self.device = device
        self.replay_buffer = replay_buffer

        # Initial observation
        self.last_observation_cpu = torch.from_numpy(self.environment.reset()).clone()
        self.last_observation = self.last_observation_cpu.to(self.device)

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    @torch.no_grad()
    def rollout(self, batch_info: BatchInfo, model: RlModel, number_of_steps: int) -> Rollout:
        """ Calculate env rollout """
        assert not model.is_recurrent, "Replay env roller does not support recurrent models"

        accumulator = TensorAccumulator()
        episode_information = []  # List of dictionaries with episode information

        for step_idx in range(number_of_steps):
            step = model.step(self.last_observation)

            replay_extra_information = {}

            accumulator.add('observations', self.last_observation_cpu)

            # Add step to the tensor accumulator
            for name, tensor in step.items():
                tensor_cpu = tensor.cpu()
                accumulator.add(name, tensor_cpu)

                if name != 'actions':
                    replay_extra_information[name] = tensor_cpu.numpy()

            actions_numpy = step['actions'].detach().cpu().numpy()
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)

            # Store rollout in the experience replay buffer
            self.replay_buffer.store_transition(
                frame=self.last_observation_cpu.numpy(),
                action=actions_numpy,
                reward=new_rewards,
                done=new_dones,
                extra_info=replay_extra_information
            )

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode

            dones_tensor = torch.from_numpy(new_dones.astype(np.float32)).clone()
            accumulator.add('dones', dones_tensor)

            self.last_observation_cpu = torch.from_numpy(new_obs).clone()
            self.last_observation = self.last_observation_cpu.to(self.device)
            accumulator.add('rewards', torch.from_numpy(new_rewards.astype(np.float32)).clone())

            episode_information.append(new_infos)

        accumulated_tensors = accumulator.result()

        return Trajectories(
            num_steps=accumulated_tensors['observations'].size(0),
            num_envs=accumulated_tensors['observations'].size(1),
            environment_information=episode_information,
            transition_tensors=accumulated_tensors,
            rollout_tensors={
                'final_values': model.value(self.last_observation).cpu()
            }
        )

    def sample(self, batch_info: BatchInfo, model: RlModel, number_of_steps: int) -> Rollout:
        """ Sample experience from replay buffer and return a batch """
        # Sample trajectories
        rollout = self.replay_buffer.sample_trajectories(rollout_length=number_of_steps, batch_info=batch_info)

        last_observations = rollout.transition_tensors['observations_next'][-1].to(self.device)
        final_values = model.value(last_observations).cpu()

        # Add 'final_values' to the rollout
        rollout.rollout_tensors['final_values'] = final_values

        return rollout

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.replay_buffer.is_ready_for_sampling()

    def update(self, rollout, batch_info):
        """ Perform update of the internal state of the buffer - e.g. for the prioritized replay weights """
        self.replay_buffer.update(rollout, batch_info)


class TrajectoryReplayEnvRollerFactory(ReplayEnvRollerFactoryBase):
    """ Factory for the ReplayEnvRoller """

    def __init__(self, replay_buffer_factory: ReplayBufferFactory):
        self.replay_buffer_factory = replay_buffer_factory

    def instantiate(self, environment, device):
        replay_buffer = self.replay_buffer_factory.instantiate(environment)

        return TrajectoryReplayEnvRoller(
            environment=environment,
            device=device,
            replay_buffer=replay_buffer
        )


def create(replay_buffer):
    """ Vel factory function """
    return TrajectoryReplayEnvRollerFactory(replay_buffer_factory=replay_buffer)
