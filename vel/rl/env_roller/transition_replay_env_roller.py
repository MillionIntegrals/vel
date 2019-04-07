import torch
import torch.nn as nn
import typing
import numpy as np

from vel.api import BatchInfo, ModelFactory
from vel.openai.baselines.common.running_mean_std import RunningMeanStd
from vel.rl.api import (
    Trajectories, Rollout, ReplayEnvRollerBase, ReplayEnvRollerFactoryBase, RlModel, ReplayBuffer, ReplayBufferFactory
)
from vel.util.tensor_accumulator import TensorAccumulator


class TransitionReplayEnvRoller(ReplayEnvRollerBase):
    """
    Calculate environment rollouts using a replay buffer for experience replay.
    Replay buffer is parametrized
    Samples transitions from the replay buffer (individual frame transitions)
    """

    def __init__(self, environment, device, replay_buffer: ReplayBuffer, discount_factor: typing.Optional[float]=None,
                 normalize_returns: bool=False, forward_steps: int=1, action_noise: typing.Optional[nn.Module]=None):
        self._environment = environment
        self.device = device
        self.replay_buffer = replay_buffer
        self.normalize_returns = normalize_returns
        self.forward_steps = forward_steps
        self.discount_factor = discount_factor
        self.action_noise = action_noise.to(self.device) if action_noise is not None else None

        if self.normalize_returns:
            assert self.discount_factor is not None, \
                "TransitionReplayEnvRoller must have a discount factor defined if normalize_returns is turned on"

        if self.forward_steps > 1:
            assert self.discount_factor is not None, \
                "TransitionReplayEnvRoller must have a discount factor defined if forward_steps is larger than one"

        self.ret_rms = RunningMeanStd(shape=()) if normalize_returns else None

        # Initial observation
        self.last_observation_cpu = torch.from_numpy(self.environment.reset()).clone()
        self.last_observation = self.last_observation_cpu.to(self.device)

        # Return normalization
        self.clip_obs = 5.0
        self.accumulated_returns = np.zeros(environment.num_envs, dtype=np.float32)

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

            if self.action_noise is not None:
                step['actions'] = self.action_noise(step['actions'], batch_info=batch_info)

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

            if self.ret_rms is not None:
                self.accumulated_returns = new_rewards + self.discount_factor * self.accumulated_returns
                self.ret_rms.update(self.accumulated_returns)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode
            dones_tensor = torch.from_numpy(new_dones.astype(np.float32)).clone()
            accumulator.add('dones', dones_tensor)

            if self.action_noise is not None:
                self.action_noise.reset_training_state(dones_tensor, batch_info=batch_info)

            self.accumulated_returns = self.accumulated_returns * (1.0 - new_dones.astype(np.float32))

            self.last_observation_cpu = torch.from_numpy(new_obs).clone()
            self.last_observation = self.last_observation_cpu.to(self.device)

            if self.ret_rms is not None:
                new_rewards = np.clip(new_rewards / np.sqrt(self.ret_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

            accumulator.add('rewards', torch.from_numpy(new_rewards.astype(np.float32)).clone())

            episode_information.append(new_infos)

        accumulated_tensors = accumulator.result()

        return Trajectories(
            num_steps=accumulated_tensors['observations'].size(0),
            num_envs=accumulated_tensors['observations'].size(1),
            environment_information=episode_information,
            transition_tensors=accumulated_tensors,
            rollout_tensors={}
        ).to_transitions()

    def sample(self, batch_info: BatchInfo, model: RlModel, number_of_steps: int) -> Rollout:
        """ Sample experience from replay buffer and return a batch """
        if self.forward_steps > 1:
            transitions = self.replay_buffer.sample_forward_transitions(
                batch_size=number_of_steps, batch_info=batch_info, forward_steps=self.forward_steps,
                discount_factor=self.discount_factor
            )
        else:
            transitions = self.replay_buffer.sample_transitions(batch_size=number_of_steps, batch_info=batch_info)

        if self.ret_rms is not None:
            rewards = transitions.transition_tensors['rewards']
            new_rewards = torch.clamp(rewards / np.sqrt(self.ret_rms.var + 1e-8), -self.clip_obs, self.clip_obs)
            transitions.transition_tensors['rewards'] = new_rewards

        return transitions

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.replay_buffer.is_ready_for_sampling()

    def initial_memory_size_hint(self) -> typing.Optional[int]:
        """ Hint how much data is needed to begin sampling, required only for diagnostics """
        return self.replay_buffer.initial_memory_size_hint()

    def update(self, rollout, batch_info):
        """ Perform update of the internal state of the buffer - e.g. for the prioritized replay weights """
        self.replay_buffer.update(rollout, batch_info)


class TransitionReplayEnvRollerFactory(ReplayEnvRollerFactoryBase):
    """ Factory for the ReplayEnvRoller """

    def __init__(self, replay_buffer_factory: ReplayBufferFactory, discount_factor: typing.Optional[float]=None,
                 normalize_returns: bool=False, forward_steps: int=1, action_noise: typing.Optional[ModelFactory]=None):
        self.replay_buffer_factory = replay_buffer_factory
        self.normalize_returns = normalize_returns
        self.forward_steps = forward_steps
        self.discount_factor = discount_factor
        self.action_noise_factory = action_noise

    def instantiate(self, environment, device):
        replay_buffer = self.replay_buffer_factory.instantiate(environment)

        if self.action_noise_factory is None:
            action_noise = None
        else:
            action_noise = self.action_noise_factory.instantiate(environment=environment)

        return TransitionReplayEnvRoller(
            environment=environment,
            device=device,
            replay_buffer=replay_buffer,
            discount_factor=self.discount_factor,
            normalize_returns=self.normalize_returns,
            forward_steps=self.forward_steps,
            action_noise=action_noise
        )


def create(replay_buffer, discount_factor: typing.Optional[float]=None, normalize_returns: bool=False,
           forward_steps: int=1, action_noise: typing.Optional[ModelFactory]=None):
    """ Vel factory function """
    return TransitionReplayEnvRollerFactory(
        replay_buffer_factory=replay_buffer,
        discount_factor=discount_factor,
        forward_steps=forward_steps,
        normalize_returns=normalize_returns,
        action_noise=action_noise
    )
