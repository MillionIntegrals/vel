import numpy as np
import sys
import torch
import tqdm

from dataclasses import dataclass

import vel.util.math as math_util

from vel.api.base import Model, ModelFactory
from vel.api.metrics import AveragingNamedMetric
from vel.api.info import EpochInfo, BatchInfo
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api.base import ReinforcerBase, ReinforcerFactory, VecEnvFactory, EnvRollerFactory, EnvRollerBase
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile, ExplainedVariance,
    EpisodeRewardMetric, FramesMetric
)


class PolicyGradientBase:
    """ Base class for policy gradient calculations """

    def initialize(self, settings):
        """ Initialize policy gradient from reinforcer settings """
        pass

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


@dataclass
class PolicyGradientSettings:
    """ Settings dataclass for a policy gradient reinforcer """
    policy_gradient: PolicyGradientBase
    number_of_steps: int
    discount_factor: float
    max_grad_norm: float = None
    gae_lambda: float = 1.0
    batch_size: int = 256
    experience_replay: int = 1


class PolicyGradientReinforcer(ReinforcerBase):
    """ Train network using a policy gradient algorithm """
    def __init__(self, device: torch.device, settings: PolicyGradientSettings, env: VecEnv, model: Model,
                 env_roller: EnvRollerBase) -> None:
        self.device = device
        self.settings = settings

        self.environment = env
        self._internal_model = model.to(self.device)

        self.env_roller = env_roller

        self.settings.policy_gradient.initialize(self.settings)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            FramesMetric("frames"),
            FPSMetric("fps"),
            EpisodeRewardMetric('PMM:episode_rewards'),
            EpisodeRewardMetricQuantile('P09:episode_rewards', quantile=0.9),
            EpisodeRewardMetricQuantile('P01:episode_rewards', quantile=0.1),
            EpisodeLengthMetric("episode_length"),
            AveragingNamedMetric("advantage_norm"),
            ExplainedVariance()
        ]

        if self.settings.max_grad_norm is not None:
            my_metrics.append(AveragingNamedMetric("grad_norm"))

        return my_metrics + self.settings.policy_gradient.metrics()

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        return self._internal_model

    def initialize_training(self):
        """ Prepare models for training """
        self.model.reset_weights()

    def train_epoch(self, epoch_info: EpochInfo) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
        for callback in epoch_info.callbacks:
            callback.on_epoch_begin(epoch_info)

        for batch_idx in tqdm.trange(epoch_info.batches_per_epoch, file=sys.stdout, desc="Training", unit="batch"):
            batch_info = BatchInfo(epoch_info, batch_idx)

            for callback in batch_info.callbacks:
                callback.on_batch_begin(batch_info)

            self.train_batch(batch_info)

            for callback in batch_info.callbacks:
                callback.on_batch_end(batch_info)

            # Even with all the experience replay, we count the single rollout as a single batch
            epoch_info.result_accumulator.calculate(batch_info)

        epoch_info.result_accumulator.freeze_results()
        epoch_info.freeze_epoch_result()

        for callback in epoch_info.callbacks:
            callback.on_epoch_end(epoch_info)

    def train_batch(self, batch_info: BatchInfo) -> None:
        """ Single, most atomic 'step' of learning this reinforcer can perform """
        # Calculate environment rollout on the evaluation version of the model
        self.model.eval()

        rollout = self.env_roller.rollout(batch_info, self.model)

        rollout_size = rollout['observations'].size(0)
        indices = np.arange(rollout_size)

        batch_splits = math_util.divide_ceiling(rollout_size, self.settings.batch_size)

        # Perform the training step
        self.model.train()

        # All policy gradient data will be put here
        batch_info['policy_gradient_data'] = []
        gradient_norms = []

        rollout_tensors = {k: v for k, v in rollout.items() if isinstance(v, torch.Tensor)}

        for i in range(self.settings.experience_replay):
            # Repeat the experience N times
            np.random.shuffle(indices)

            for sub_indices in np.array_split(indices, batch_splits):
                batch_rollout = {k: v[sub_indices] for k, v in rollout_tensors.items()}

                batch_info.optimizer.zero_grad()

                loss = self.settings.policy_gradient.calculate_loss(
                    batch_info=batch_info,
                    device=self.device,
                    model=self.model,
                    rollout=batch_rollout,
                )

                loss.backward()

                # Gradient clipping
                if self.settings.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        max_norm=self.settings.max_grad_norm
                    )

                    gradient_norms.append(grad_norm)

                batch_info.optimizer.step(closure=None)

        batch_info['frames'] = torch.tensor(rollout_size).to(self.device)
        batch_info['episode_infos'] = rollout['episode_information']
        batch_info['advantage_norm'] = torch.norm(rollout['advantages'])
        batch_info['values'] = rollout['values']
        batch_info['rewards'] = rollout['discounted_rewards']
        batch_info['grad_norm'] = torch.tensor(np.mean(gradient_norms)).to(self.device)

        # Put in aggregated
        data_dict_keys = {y for x in batch_info['policy_gradient_data'] for y in x.keys()}

        for key in data_dict_keys:
            # Just average all the statistics from the loss function
            batch_info[key] = torch.mean(torch.stack([d[key] for d in batch_info['policy_gradient_data']]))


class PolicyGradientReinforcerFactory(ReinforcerFactory):
    """ Vel factory class for the PolicyGradientReinforcer """
    def __init__(self, settings, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 env_roller_factory: EnvRollerFactory, parallel_envs: int, seed: int):
        self.settings = settings

        self.env_roller_factory = env_roller_factory
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.parallel_envs = parallel_envs
        self.seed = seed

    def instantiate(self, device: torch.device) -> ReinforcerBase:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        model = self.model_factory.instantiate(action_space=env.action_space)
        env_roller = self.env_roller_factory.instantiate(environment=env, device=device, settings=self.settings)

        return PolicyGradientReinforcer(device, self.settings, env, model, env_roller)
