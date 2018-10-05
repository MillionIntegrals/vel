import attr
import numpy as np
import sys
import torch
import tqdm

import vel.util.math as math_util

from vel.api.base import Model, ModelFactory
from vel.api.info import EpochInfo, BatchInfo
from vel.rl.api.base import ReinforcerBase, ReinforcerFactory, VecEnvFactory, EnvRollerFactory, EnvRollerBase, AlgoBase
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile,
    EpisodeRewardMetric, FramesMetric
)


@attr.s(auto_attribs=True)
class OnPolicyIterationReinforcerSettings:
    """ Settings dataclass for a policy gradient reinforcer """
    discount_factor: float
    number_of_steps: int
    batch_size: int = 256
    experience_replay: int = 1


class OnPolicyIterationReinforcer(ReinforcerBase):
    """
    A reinforcer that calculates on-policy environment rollouts and uses them to train policy directly.
    May split the sample into multiple batches and may replay batches a few times.
    """
    def __init__(self, device: torch.device, settings: OnPolicyIterationReinforcerSettings, model: Model,
                 algo: AlgoBase, env_roller: EnvRollerBase) -> None:
        self.device = device
        self.settings = settings

        self._trained_model = model.to(self.device)

        self.env_roller = env_roller
        self.algo = algo

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length"),
        ]

        return my_metrics + self.algo.metrics() + self.env_roller.metrics()

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        return self._trained_model

    def initialize_training(self, training_info):
        """ Prepare models for training """
        self.model.reset_weights()
        self.algo.initialize(
            self.settings, model=self.model, environment=self.env_roller.environment, device=self.device
        )

    def train_epoch(self, epoch_info: EpochInfo) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
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

        1. Roll out the environmnent using current policy
        2. Use that rollout to train the policy
        """
        # Calculate environment rollout on the evaluation version of the model
        self.model.eval()

        rollout = self.env_roller.rollout(batch_info, self.model)

        rollout_size = rollout['size']
        indices = np.arange(rollout_size)

        # We may potentially need to split rollout into multiple batches
        batch_splits = math_util.divide_ceiling(rollout_size, self.settings.batch_size)

        # Perform the training step
        self.model.train()

        # Algo will aggregate data into this list:
        batch_info['sub_batch_data'] = []

        rollout_tensors = {k: v for k, v in rollout.items() if isinstance(v, torch.Tensor)}

        for i in range(self.settings.experience_replay):
            # Repeat the experience N times
            np.random.shuffle(indices)

            for sub_indices in np.array_split(indices, batch_splits):
                batch_rollout = {k: v[sub_indices] for k, v in rollout_tensors.items()}

                batch_result = self.algo.optimizer_step(
                    batch_info=batch_info,
                    device=self.device,
                    model=self.model,
                    rollout=batch_rollout
                )

                batch_info['sub_batch_data'].append(batch_result)

        batch_info['frames'] = rollout_size
        batch_info['episode_infos'] = rollout['episode_information']

        # Even with all the experience replay, we count the single rollout as a single batch
        batch_info.aggregate_key('sub_batch_data')


class PolicyGradientReinforcerFactory(ReinforcerFactory):
    """ Vel factory class for the PolicyGradientReinforcer """
    def __init__(self, settings, parallel_envs: int, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 algo: AlgoBase, env_roller_factory: EnvRollerFactory, seed: int):
        self.settings = settings
        self.parallel_envs = parallel_envs

        self.env_factory = env_factory
        self.model_factory = model_factory
        self.algo = algo
        self.env_roller_factory = env_roller_factory
        self.seed = seed

    def instantiate(self, device: torch.device) -> ReinforcerBase:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        model = self.model_factory.instantiate(action_space=env.action_space)
        env_roller = self.env_roller_factory.instantiate(environment=env, device=device, settings=self.settings)

        return OnPolicyIterationReinforcer(device, self.settings, model, self.algo, env_roller)


def create(model_config, model, vec_env, algo, env_roller, parallel_envs, number_of_steps,
           discount_factor, batch_size=256, experience_replay=1):
    """ Create a policy gradient reinforcer - factory """
    settings = OnPolicyIterationReinforcerSettings(
        discount_factor=discount_factor,
        number_of_steps=number_of_steps,
        batch_size=batch_size,
        experience_replay=experience_replay
    )

    return PolicyGradientReinforcerFactory(
        settings=settings,
        parallel_envs=parallel_envs,
        env_factory=vec_env,
        model_factory=model,
        algo=algo,
        env_roller_factory=env_roller,
        seed=model_config.seed
    )
