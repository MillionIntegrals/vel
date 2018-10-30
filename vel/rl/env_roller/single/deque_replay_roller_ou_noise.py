import numpy as np
import torch

from vel.math.processes import OrnsteinUhlenbeckNoiseProcess
from vel.openai.baselines.common.running_mean_std import RunningMeanStd
from vel.rl.api import Rollout, Transitions
from vel.rl.api.base import ReplayEnvRollerBase, ReplayEnvRollerFactory
from vel.rl.buffers.deque_backend import DequeBufferBackend


class DequeReplayRollerOuNoise(ReplayEnvRollerBase):
    """
    Enrionment roller with experience replay buffer rolling out a **single** environment
    with Ornstein–Uhlenbeck noise process
    """

    def __init__(self, environment, device, batch_size, buffer_capacity, buffer_initial_size, noise_std_dev,
                 discount_factor, normalize_observations=False, normalize_returns=False):
        self.device = device
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.discount_factor = discount_factor

        self.device = device
        self._environment = environment

        self.backend = DequeBufferBackend(
            buffer_capacity=self.buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

        self.last_observation = self.environment.reset()

        len_action_space = self.environment.action_space.shape[-1]

        self.noise_process = OrnsteinUhlenbeckNoiseProcess(
            np.zeros(len_action_space), float(noise_std_dev) * np.ones(len_action_space)
        )

        self.ob_rms = RunningMeanStd(shape=self.environment.observation_space.shape) if normalize_observations else None
        self.ret_rms = RunningMeanStd(shape=()) if normalize_returns else None
        self.clip_obs = 5.0
        self.accumulated_return = 0.0

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.backend.current_size >= self.buffer_initial_size

    def _observation_to_tensor(self, observation_array):
        """ Convert observation numpy array to a tensor """
        return torch.from_numpy(self._filter_observation(observation_array)).to(self.device)

    @torch.no_grad()
    def rollout(self, batch_info, model) -> Rollout:
        """ Roll-out the environment and return it """
        observation_tensor = self._observation_to_tensor(self.last_observation[None])

        step = model.step(observation_tensor)
        action = step['actions'].detach().cpu().numpy()[0]
        noise = self.noise_process()

        action_perturbed = np.clip(
            action + noise, self.environment.action_space.low, self.environment.action_space.high
        )

        observation, reward, done, info = self.environment.step(action_perturbed)

        if self.ob_rms is not None:
            self.ob_rms.update(observation[None])

        if self.ret_rms is not None:
            self.accumulated_return = reward + self.discount_factor * self.accumulated_return

            self.ret_rms.update(np.array([self.accumulated_return]))

        self.backend.store_transition(self.last_observation, action_perturbed, reward, done)

        # Usual, reset on done
        if done:
            observation = self.environment.reset()
            self.noise_process.reset()
            self.accumulated_return = 0.0

        self.last_observation = observation

        return Transitions(
            size=1,
            environment_information=[info],
            transition_tensors={
                'actions': step['actions'],
                'values': step['values']
            },
        )

    def _filter_observation(self, obs):
        """ Potentially normalize observation """
        if self.ob_rms is not None:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

            return obs.astype(np.float32)
        else:
            return obs

    def sample(self, batch_info, model) -> Transitions:
        """ Sample experience from replay buffer and return a batch """
        indexes = self.backend.sample_batch_uniform(self.batch_size, history_length=1)
        batch = self.backend.get_batch(indexes, history_length=1)

        observations = self._observation_to_tensor(batch['states'])
        observations_plus1 = self._observation_to_tensor(batch['states+1'])

        rewards = batch['rewards'].astype(np.float32)

        if self.ret_rms is not None:
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

        dones = torch.from_numpy(batch['dones'].astype(np.float32)).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)

        return Transitions(
            size=self.batch_size,
            environment_information=[],
            transition_tensors={
                'observations': observations,
                'observations_next': observations_plus1,
                'dones': dones,
                'rewards': rewards,
                'actions': actions
            }
        )


class DequeReplayRollerOuNoiseFactory(ReplayEnvRollerFactory):
    """ Factory class for DequeReplayQRoller """
    def __init__(self, buffer_capacity: int, buffer_initial_size: int, noise_std_dev: float,
                 normalize_observations: bool=False, normalize_returns: bool=False):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.noise_std_dev = noise_std_dev
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns

    def instantiate(self, environment, device, settings) -> ReplayEnvRollerBase:
        return DequeReplayRollerOuNoise(
            environment=environment,
            device=device,
            batch_size=settings.batch_size,
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            discount_factor=settings.discount_factor,
            noise_std_dev=self.noise_std_dev,
            normalize_observations=self.normalize_observations,
            normalize_returns=self.normalize_returns
        )


def create(buffer_capacity: int, buffer_initial_size: int, noise_std_dev: float,
           normalize_observations=False, normalize_returns=False):
    return DequeReplayRollerOuNoiseFactory(
        noise_std_dev=noise_std_dev,
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        normalize_observations=normalize_observations,
        normalize_returns=normalize_returns
    )
