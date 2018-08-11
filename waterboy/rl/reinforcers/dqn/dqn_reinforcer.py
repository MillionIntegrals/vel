import tqdm
import sys
import numpy as np

import gym
import torch
import torch.nn.functional as F

from torch.optim import Optimizer

from dataclasses import dataclass

from waterboy.api import BatchIdx, EpochIdx
from waterboy.api.base import Model, ModelFactory, Schedule
from waterboy.api.metrics import EpochResultAccumulator
from waterboy.api.metrics.averaging_metric import AveragingNamedMetric
from waterboy.api.metrics.summing_metric import SummingNamedMetric
from waterboy.rl.api.base import ReinforcerBase, ReinforcerFactory, EnvFactory
from waterboy.rl.metrics import FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile, EpisodeRewardMetric


class DqnBufferBase:
    """ Base class for DQN buffers """
    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        raise NotImplementedError

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float):
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        raise NotImplementedError

    def sample(self, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
        raise NotImplementedError

    def is_ready(self):
        """ If buffer is ready for training """
        raise NotImplementedError

    # def store_frame(self, frame):
    #     """ Add another frame to the buffer """
    #     raise NotImplementedError
    #
    # def store_transition(self, action, reward, done):
    #     """ Add frame transition to the buffer """
    #     raise NotImplementedError
    #
    # def get_frame(self, idx):
    #     """ Get frame with given IDX """
    #     raise NotImplementedError
    #
    # def sample(self, batch_size):
    #     """ Calculate random sample from the replay buffer """
    #     raise NotImplementedError


@dataclass
class DqnReinforcerSettings:
    """ Settings class for deep Q-Learning """
    buffer: DqnBufferBase
    epsilon_schedule: Schedule

    train_frequency: int
    batch_size: int

    target_update_frequency: int

    discount_factor: float
    max_grad_norm: float
    seed: int


class DqnReinforcer(ReinforcerBase):
    """
    Implementation of Deep Q-Learning from DeepMinds Nature paper
    "Human-level control through deep reinforcement learning"
    """
    def __init__(self, device, settings: DqnReinforcerSettings, environment: gym.Env, train_model: Model, target_model: Model):
        self.device = device
        self.settings = settings
        self.environment = environment

        self.train_model = train_model.to(self.device)
        self.target_model = target_model.to(self.device)

        self.buffer = self.settings.buffer

        self.buffer.initialize(self.environment, self.device)
        self.last_observation = self.environment.reset()

    def metrics(self) -> list:
        return [
            SummingNamedMetric("frames", reset_value=False),
        ]

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            SummingNamedMetric("frames", reset_value=False),
            FPSMetric(),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric(),
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

    def train_epoch(self, epoch_idx: EpochIdx, batches_per_epoch: int, optimizer: Optimizer, callbacks: list,
                    result_accumulator: EpochResultAccumulator = None) -> dict:
        for callback in callbacks:
            callback.on_epoch_begin(epoch_idx)

        for batch_idx_number in tqdm.trange(batches_per_epoch, file=sys.stdout, desc="Training", unit="batch"):
            extra = {}

            if 'total_frames' in epoch_idx.extra:
                extra['progress_meter'] = (
                        result_accumulator.intermediate_value('frames') / epoch_idx.extra['total_frames']
                )

            progress_idx = BatchIdx(epoch_idx, batch_idx_number, batches_per_epoch=batches_per_epoch, extra=extra)

            for callback in callbacks:
                callback.on_batch_begin(progress_idx)

            self.train_step(progress_idx, optimizer, result_accumulator)

            for callback in callbacks:
                callback.on_batch_end(progress_idx, result_accumulator.value(), optimizer)

        result_accumulator.freeze_results()

        epoch_result = result_accumulator.result()

        for callback in callbacks:
            callback.on_epoch_end(epoch_idx, epoch_result)

        return epoch_result

    def buffer_rollout(self, epsilon_value):
        """ Evaluate environment and store in the buffer """
        # This probably should be moved to the buffer?

    def train_step(self, batch_idx: BatchIdx, optimizer: Optimizer,
                   result_accumulator: EpochResultAccumulator = None) -> None:
        # Each DQN batch is
        # 1. Prepare everything
        self.model.eval()
        self.target_model.eval()
        data_dict = {}
        episode_information = []

        # 2. Choose and evaluate actions, roll out env
        # For the whole initialization epsilon will stay fixed, because the network is not learning either way
        epsilon_value = self.settings.epsilon_schedule.value(batch_idx.extra['progress_meter'])

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
        optimizer.zero_grad()

        batch_sample = self.buffer.sample(self.settings.batch_size)

        observation_tensor = torch.from_numpy(batch_sample['observations']).to(self.device)
        observation_tensor_tplus1 = torch.from_numpy(batch_sample['observations_tplus1']).to(self.device)
        dones_tensor = torch.from_numpy(batch_sample['dones'].astype(np.float32)).to(self.device)
        rewards_tensor = torch.from_numpy(batch_sample['rewards'].astype(np.float32)).to(self.device)

        actions_tensor = torch.from_numpy(batch_sample['actions']).to(self.device)
        one_hot_actions = torch.eye(self.environment.action_space.n, device=self.device)[actions_tensor]

        with torch.no_grad():
            values = self.target_model(observation_tensor_tplus1).max(dim=1)[0]
            expected_q = rewards_tensor + self.settings.discount_factor * values * (1 - dones_tensor.float())

        q = self.model(observation_tensor)
        q_selected = (q * one_hot_actions).sum(dim=1)

        loss = F.smooth_l1_loss(q_selected, expected_q.detach())
        loss.backward()

        if self.settings.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=self.settings.max_grad_norm
            )

            data_dict['grad_norm'] = torch.tensor(grad_norm).to(self.device)

        optimizer.step(closure=None)

        data_dict['frames'] = torch.tensor(frames).to(self.device)
        data_dict['episode_infos'] = episode_information
        data_dict['loss'] = loss
        data_dict['epsilon'] = torch.tensor(epsilon_value).to(self.device)
        data_dict['average_q_selected'] = torch.mean(q_selected)
        data_dict['average_q_target'] = torch.mean(expected_q)

        if result_accumulator is not None:
            result_accumulator.calculate(data_dict)

        if batch_idx.aggregate_batch_number % self.settings.target_update_frequency:
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
