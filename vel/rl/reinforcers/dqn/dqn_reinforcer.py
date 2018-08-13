import tqdm
import sys
import numpy as np

from dataclasses import dataclass

import gym
import torch
import torch.nn.functional as F

from vel.api import BatchInfo, EpochInfo
from vel.api.base import Model, ModelFactory, Schedule
from vel.api.metrics import AveragingNamedMetric
from vel.rl.api.base import ReinforcerBase, ReinforcerFactory, EnvFactory
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile, EpisodeRewardMetric, FramesMetric
)


class DqnBufferBase:
    """ Base class for DQN buffers """
    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        raise NotImplementedError

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float) -> dict:
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        raise NotImplementedError

    def sample(self, batch_info, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
        raise NotImplementedError

    def is_ready(self) -> bool:
        """ If buffer is ready for training """
        raise NotImplementedError

    def update(self, sample, errors):
        """ If buffer is ready for training """
        pass


@dataclass
class DqnReinforcerSettings:
    """ Settings class for deep Q-Learning """
    buffer: DqnBufferBase
    epsilon_schedule: Schedule

    train_frequency: int
    batch_size: int
    double_dqn: bool

    target_update_frequency: int

    discount_factor: float
    max_grad_norm: float
    seed: int


class DqnReinforcer(ReinforcerBase):
    """
    Implementation of Deep Q-Learning from DeepMinds Nature paper
    "Human-level control through deep reinforcement learning"
    """
    def __init__(self, device, settings: DqnReinforcerSettings, environment: gym.Env, train_model: Model,
                 target_model: Model):
        self.device = device
        self.settings = settings
        self.environment = environment

        self.train_model = train_model.to(self.device)
        self.target_model = target_model.to(self.device)

        self.buffer = self.settings.buffer

        self.buffer.initialize(self.environment, self.device)
        self.last_observation = self.environment.reset()

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length"),
            AveragingNamedMetric("loss"),
            AveragingNamedMetric("epsilon"),
            AveragingNamedMetric("average_q_selected"),
            AveragingNamedMetric("average_q_target"),
        ]

        if self.settings.max_grad_norm is not None:
            my_metrics.append(AveragingNamedMetric("grad_norm"))

        return my_metrics

    @property
    def model(self) -> Model:
        return self.train_model

    def initialize_training(self):
        """ Prepare models for training """
        self.train_model.reset_weights()
        self.target_model.load_state_dict(self.target_model.state_dict())

    def train_epoch(self, epoch_info: EpochInfo) -> None:
        for callback in epoch_info.callbacks:
            callback.on_epoch_begin(epoch_info)

        for batch_idx in tqdm.trange(epoch_info.batches_per_epoch, file=sys.stdout, desc="Training", unit="batch"):
            batch_info = BatchInfo(epoch_info, batch_idx)

            for callback in batch_info.callbacks:
                callback.on_batch_begin(batch_info)

            self.train_batch(batch_info)

            for callback in batch_info.callbacks:
                callback.on_batch_end(batch_info)

            epoch_info.result_accumulator.calculate(batch_info)

        epoch_info.result_accumulator.freeze_results()
        epoch_info.freeze_epoch_result()

        for callback in epoch_info.callbacks:
            callback.on_epoch_end(epoch_info)

    def train_batch(self, batch_info: BatchInfo) -> None:
        # Each DQN batch is
        # 1. Prepare everything
        self.model.eval()
        self.target_model.eval()

        episode_information = []

        # 2. Choose and evaluate actions, roll out env
        # For the whole initialization epsilon will stay fixed, because the network is not learning either way
        epsilon_value = self.settings.epsilon_schedule.value(batch_info['progress'])

        frames = 0

        with torch.no_grad():
            if not self.buffer.is_ready():
                while not self.buffer.is_ready():
                    maybe_episode_info = self.buffer.rollout(self.environment, self.model, epsilon_value)

                    if maybe_episode_info is not None:
                        episode_information.append(maybe_episode_info)

                    frames += 1
            else:
                for i in range(self.settings.train_frequency):
                    maybe_episode_info = self.buffer.rollout(self.environment, self.model, epsilon_value)

                    if maybe_episode_info is not None:
                        episode_information.append(maybe_episode_info)

                    frames += 1

        # 2. Perform experience replay and train the network
        self.model.train()
        batch_info.optimizer.zero_grad()

        batch_sample = self.buffer.sample(batch_info, self.settings.batch_size)

        observation_tensor = torch.from_numpy(batch_sample['observations']).to(self.device)
        observation_tensor_tplus1 = torch.from_numpy(batch_sample['observations_tplus1']).to(self.device)
        dones_tensor = torch.from_numpy(batch_sample['dones'].astype(np.float32)).to(self.device)
        rewards_tensor = torch.from_numpy(batch_sample['rewards'].astype(np.float32)).to(self.device)

        actions_tensor = torch.from_numpy(batch_sample['actions']).to(self.device)

        with torch.no_grad():
            if self.settings.double_dqn:
                # DOUBLE DQN
                target_values = self.target_model(observation_tensor_tplus1)
                model_values = self.model(observation_tensor_tplus1)
                # Select largest 'target' value based on action that 'model' selects
                values = target_values.gather(1, model_values.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                # REGULAR DQN
                values = self.target_model(observation_tensor_tplus1).max(dim=1)[0]

            expected_q = rewards_tensor + self.settings.discount_factor * values * (1 - dones_tensor.float())

        q = self.model(observation_tensor)
        q_selected = q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        original_losses = F.smooth_l1_loss(q_selected, expected_q.detach(), reduction='none')

        weights_tensor = torch.from_numpy(batch_sample['weights']).float().to(self.device)
        loss = torch.mean(weights_tensor * original_losses)

        loss.backward()

        self.buffer.update(batch_sample, original_losses)

        if self.settings.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=self.settings.max_grad_norm
            )

            batch_info['grad_norm'] = torch.tensor(grad_norm).to(self.device)

        batch_info.optimizer.step(closure=None)

        batch_info['frames'] = torch.tensor(frames).to(self.device)
        batch_info['episode_infos'] = episode_information
        batch_info['loss'] = loss
        batch_info['epsilon'] = torch.tensor(epsilon_value).to(self.device)
        batch_info['average_q_selected'] = torch.mean(q_selected)
        batch_info['average_q_target'] = torch.mean(expected_q)

        if batch_info.aggregate_batch_number % self.settings.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())


class DqnReinforcerFactory(ReinforcerFactory):
    """ Factory class for the DQN reinforcer """

    def __init__(self, settings, env_factory: EnvFactory, model_factory: ModelFactory):
        self.settings = settings

        self.env_factory = env_factory
        self.model_factory = model_factory

    def instantiate(self, device: torch.device) -> DqnReinforcer:
        env = self.env_factory.instantiate(seed=self.settings.seed)

        train_model = self.model_factory.instantiate(action_space=env.action_space)
        target_model = self.model_factory.instantiate(action_space=env.action_space)
        return DqnReinforcer(device, self.settings, env, train_model, target_model)
