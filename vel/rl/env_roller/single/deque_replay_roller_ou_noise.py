import numpy as np
import torch

from vel.math.processes import OrnsteinUhlenbeckNoiseProcess
from vel.openai.baselines.common.running_mean_std import RunningMeanStd
from vel.rl.api.base import ReplayEnvRollerBase, ReplayEnvRollerFactory
from vel.rl.buffers.deque_backend import DequeBufferBackend


class DequeReplayRollerOuNoise(ReplayEnvRollerBase):
    """
    Enrionment roller with experience replay buffer rolling out a **single** environment
    with Ornstein–Uhlenbeck noise process
    """

    def __init__(self, environment, device, batch_size, buffer_capacity, buffer_initial_size, noise_std_dev,
                 normalize_observations=False):
        self.device = device
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.normalize_observations = normalize_observations

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
        self.clip_obs = 10.0

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.backend.current_size >= self.buffer_initial_size

    def rollout(self, batch_info, model) -> dict:
        """ Roll-out the environment and return it """
        observation_tensor = torch.from_numpy(self.last_observation).to(self.device)

        step = model.step(observation_tensor[None])
        action = step['actions'].detach().cpu().numpy()[0]
        noise = self.noise_process()

        action_perturbed = np.clip(
            action + noise, self.environment.action_space.low, self.environment.action_space.high
        )

        observation, reward, done, info = self.environment.step(action_perturbed)

        if self.ob_rms is not None:
            self.ob_rms.update(observation)

        self.backend.store_transition(self.last_observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = self.environment.reset()
            self.noise_process.reset()

        self.last_observation = observation

        return {
            'episode_information': info.get('episode'),
            'action': step['actions'][0],
            'value': step['values'][0]
        }

    def _filter_observation(self, obs):
        """ Potentially normalize observation """
        if self.ob_rms is not None:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

            return obs.astype(np.float32)
        else:
            return obs

    def sample(self, batch_info, model) -> dict:
        """ Sample experience from replay buffer and return a batch """
        indexes = self.backend.sample_batch_uniform(self.batch_size, 1)
        batch = self.backend.get_batch(indexes, 1)

        observations = torch.from_numpy(self._filter_observation(batch['states'])).to(self.device)
        observations_plus1 = torch.from_numpy(self._filter_observation(batch['states+1'])).to(self.device)
        dones = torch.from_numpy(batch['dones'].astype(np.float32)).to(self.device)
        rewards = torch.from_numpy(batch['rewards'].astype(np.float32)).to(self.device)
        actions = torch.from_numpy(batch['actions']).to(self.device)

        return {
            'size': self.batch_size,
            'observations': observations,
            'observations+1': observations_plus1,
            'dones': dones,
            'rewards': rewards,
            'actions': actions
        }


class DequeReplayRollerOuNoiseFactory(ReplayEnvRollerFactory):
    """ Factory class for DequeReplayQRoller """
    def __init__(self, buffer_capacity: int, buffer_initial_size: int, noise_std_dev: float,
                 normalize_observations: bool=False):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.noise_std_dev = noise_std_dev
        self.normalize_observations = normalize_observations

    def instantiate(self, environment, device, settings) -> ReplayEnvRollerBase:
        return DequeReplayRollerOuNoise(
            environment=environment,
            device=device,
            batch_size=settings.batch_size,
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            noise_std_dev=self.noise_std_dev,
            normalize_observations=self.normalize_observations
        )


def create(buffer_capacity: int, buffer_initial_size: int, noise_std_dev: float,
           normalize_observations=False):
    return DequeReplayRollerOuNoiseFactory(
        noise_std_dev=noise_std_dev,
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        normalize_observations=normalize_observations
    )
