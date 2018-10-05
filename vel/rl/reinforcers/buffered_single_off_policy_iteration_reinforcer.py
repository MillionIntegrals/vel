import attr
import numpy as np
import sys
import tqdm

import gym
import torch

from vel.api import BatchInfo, EpochInfo
from vel.api.base import Model, ModelFactory
from vel.api.metrics import AveragingNamedMetric
from vel.rl.api.base import ReinforcerBase, ReinforcerFactory, EnvFactory, ReplayEnvRollerBase, AlgoBase
from vel.rl.api.base.env_roller import ReplayEnvRollerFactory
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile, EpisodeRewardMetric, FramesMetric,
)


@attr.s(auto_attribs=True)
class BufferedSingleOffPolicyIterationReinforcerSettings:
    """ Settings class for deep Q-Learning """
    batch_rollout_rounds: int
    batch_training_rounds: int

    batch_size: int
    discount_factor: float


class BufferedSingleOffPolicyIterationReinforcer(ReinforcerBase):
    """
    An off-policy reinforcer that rolls out **single** environment and stores transitions in a buffer.
    Afterwards, it samples batches experience from this buffer to train the policy.
    """
    def __init__(self, device: torch.device, settings: BufferedSingleOffPolicyIterationReinforcerSettings,
                 environment: gym.Env, model: Model, algo: AlgoBase, env_roller: ReplayEnvRollerBase):
        self.device = device
        self.settings = settings
        self.environment = environment

        self._trained_model = model.to(self.device)
        self.algo = algo

        self.env_roller = env_roller

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length"),
            AveragingNamedMetric("rollout_action_mean"),
            AveragingNamedMetric("rollout_action_std"),
            AveragingNamedMetric("rollout_value_mean")
        ]

        return my_metrics + self.algo.metrics() + self.env_roller.metrics()

    @property
    def model(self) -> Model:
        return self._trained_model

    def initialize_training(self, training_info):
        """ Prepare models for training """
        self.model.reset_weights()
        self.algo.initialize(self.settings, model=self.model, environment=self.environment, device=self.device)

    def train_epoch(self, epoch_info: EpochInfo) -> None:
        """ Train model for a single epoch  """
        epoch_info.on_epoch_begin()

        for batch_idx in tqdm.trange(epoch_info.batches_per_epoch, file=sys.stdout, desc="Training", unit="batch"):
            batch_info = BatchInfo(epoch_info, batch_idx)

            batch_info.on_batch_begin()
            self.train_batch(batch_info)
            batch_info.on_batch_end()

        epoch_info.result_accumulator.freeze_results()
        epoch_info.on_epoch_end()

    def train_batch(self, batch_info: BatchInfo) -> None:
        """
        Batch - the most atomic unit of learning.

        For this reinforforcer, that involves:

        1. Roll out environment and store out experience in the buffer
        2. Sample the buffer and train the algo on sample batch
        """
        # Each DQN batch is
        # 1. Roll out environment and store out experience in the buffer
        self.model.eval()

        # Helper variables for rollouts
        episode_information = []
        rollout_actions = []
        rollout_values = []
        frames = 0

        with torch.no_grad():
            if not self.env_roller.is_ready_for_sampling():
                while not self.env_roller.is_ready_for_sampling():
                    rollout = self.env_roller.rollout(batch_info, self.model)
                    maybe_episode_info = rollout['episode_information']

                    if maybe_episode_info is not None:
                        episode_information.append(maybe_episode_info)

                    frames += 1
                    rollout_actions.append(rollout['action'].detach().cpu().numpy())
                    rollout_values.append(rollout['value'].detach().cpu().numpy())
            else:
                for i in range(self.settings.batch_rollout_rounds):
                    rollout = self.env_roller.rollout(batch_info, self.model)
                    maybe_episode_info = rollout['episode_information']

                    if maybe_episode_info is not None:
                        episode_information.append(maybe_episode_info)

                    frames += 1
                    rollout_actions.append(rollout['action'].detach().cpu().numpy())
                    rollout_values.append(rollout['value'].detach().cpu().numpy())

        batch_info['rollout_action_mean'] = np.mean(rollout_actions)
        batch_info['rollout_action_std'] = np.std(rollout_actions)
        batch_info['rollout_value_mean'] = np.std(rollout_values)

        batch_info['frames'] = frames
        batch_info['episode_infos'] = episode_information

        # 2. Sample the buffer and train the algo on sample batch
        self.model.train()

        # Algo will aggregate data into this list:
        batch_info['sub_batch_data'] = []

        for i in range(self.settings.batch_training_rounds):
            batch_sample = self.env_roller.sample(batch_info, self.model)

            batch_result = self.algo.optimizer_step(
                batch_info=batch_info,
                device=self.device,
                model=self.model,
                rollout=batch_sample
            )

            self.env_roller.update(sample=batch_sample, batch_info=batch_result)

            batch_info['sub_batch_data'].append(batch_result)

        batch_info.aggregate_key('sub_batch_data')


class BufferedSingleOffPolicyIterationReinforcerFactory(ReinforcerFactory):
    """ Factory class for the DQN reinforcer """

    def __init__(self, settings, env_factory: EnvFactory, model_factory: ModelFactory,
                 algo: AlgoBase, env_roller_factory: ReplayEnvRollerFactory, seed: int):
        self.settings = settings

        self.env_factory = env_factory
        self.model_factory = model_factory
        self.algo = algo
        self.env_roller_factory = env_roller_factory
        self.seed = seed

    def instantiate(self, device: torch.device) -> BufferedSingleOffPolicyIterationReinforcer:
        env = self.env_factory.instantiate(seed=self.seed)
        env_roller = self.env_roller_factory.instantiate(env, device, self.settings)
        model = self.model_factory.instantiate(action_space=env.action_space)

        return BufferedSingleOffPolicyIterationReinforcer(
            device=device,
            settings=self.settings,
            environment=env,
            model=model,
            algo=self.algo,
            env_roller=env_roller
        )


def create(model_config, env, model, algo, env_roller, batch_size: int, discount_factor: float,
           batch_rollout_rounds=1, batch_training_rounds=1):
    """ Vel creation function for DqnReinforcerFactory """
    settings = BufferedSingleOffPolicyIterationReinforcerSettings(
        batch_rollout_rounds=batch_rollout_rounds,
        batch_training_rounds=batch_training_rounds,
        batch_size=batch_size,
        discount_factor=discount_factor
    )

    return BufferedSingleOffPolicyIterationReinforcerFactory(
        settings=settings,
        env_factory=env,
        model_factory=model,
        algo=algo,
        env_roller_factory=env_roller,
        seed=model_config.seed
    )
