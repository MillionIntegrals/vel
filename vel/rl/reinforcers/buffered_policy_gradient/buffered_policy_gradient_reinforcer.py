import sys
import torch
import tqdm
import numpy as np

from dataclasses import dataclass

from vel.api import EpochInfo, BatchInfo
from vel.api.base import Model, ModelFactory
from vel.api.metrics import AveragingNamedMetric
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api.base import ReinforcerBase, ReinforcerFactory, VecEnvFactory, ReplayEnvRollerBase
from vel.rl.api.base.env_roller import ReplayEnvRollerFactory
from vel.rl.metrics import (
    FPSMetric, EpisodeLengthMetric, EpisodeRewardMetricQuantile,
    EpisodeRewardMetric, FramesMetric
)
from vel.rl.reinforcers.policy_gradient.policy_gradient_reinforcer import PolicyGradientBase


@dataclass
class BufferedPolicyGradientSettings:
    """ Settings dataclass for a policy gradient reinforcer """
    number_of_steps: int
    discount_factor: float
    max_grad_norm: float = None
    gae_lambda: float = 1.0
    batch_size: int = 256
    experience_replay: int = 1
    stochastic_experience_replay: bool = True


class BufferedPolicyGradientReinforcer(ReinforcerBase):
    """ Factory class replay buffer reinforcer """

    def __init__(self, device: torch.device, settings: BufferedPolicyGradientSettings, env: VecEnv, model: Model,
                 env_roller: ReplayEnvRollerBase, policy_gradient: PolicyGradientBase) -> None:
        self.device = device
        self.settings = settings

        self.environment = env
        self._internal_model = model.to(self.device)

        self.env_roller = env_roller
        self.policy_gradient = policy_gradient

        self.policy_gradient.initialize(self.settings)

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

        if self.settings.max_grad_norm is not None:
            my_metrics.append(AveragingNamedMetric("grad_norm"))

        return my_metrics + self.policy_gradient.metrics()

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        return self._internal_model

    def initialize_training(self):
        """ Prepare models for training """
        self.model.reset_weights()

    def train_epoch(self, epoch_info: EpochInfo):
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

    def train_batch(self, batch_info: BatchInfo):
        """ Single, most atomic 'step' of learning this reinforcer can perform """
        batch_info['gradient_norms'] = []
        batch_info['policy_gradient_data'] = []

        self.on_policy_train_batch(batch_info)

        if self.settings.experience_replay > 0 and self.env_roller.is_ready_for_sampling():
            if self.settings.stochastic_experience_replay:
                experience_replay_count = np.random.poisson(self.settings.experience_replay)
            else:
                experience_replay_count = self.settings.experience_replay

            for i in range(experience_replay_count):
                self.off_policy_train_batch(batch_info)

        if self.settings.max_grad_norm is not None:
            batch_info['grad_norm'] = torch.tensor(np.mean(batch_info['gradient_norms'])).to(self.device)

        # Aggregate policy gradient data
        data_dict_keys = {y for x in batch_info['policy_gradient_data'] for y in x.keys()}

        for key in data_dict_keys:
            # Just average all the statistics from the loss function
            batch_info[key] = torch.mean(torch.stack([d[key] for d in batch_info['policy_gradient_data']]))

    def on_policy_train_batch(self, batch_info: BatchInfo):
        """ Perform an 'on-policy' training step of evaluating an env and a single backpropagation step """
        self.model.eval()

        rollout = self.env_roller.rollout(batch_info, self.model)

        self.model.train()
        batch_info.optimizer.zero_grad()

        loss = self.policy_gradient.calculate_loss(
            batch_info=batch_info,
            device=self.device,
            model=self.model,
            rollout=rollout
        )

        loss.backward()

        # Gradient clipping
        if self.settings.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=self.settings.max_grad_norm
            )

            batch_info['gradient_norms'].append(grad_norm)

        batch_info.optimizer.step(closure=None)

        batch_info['frames'] = torch.tensor(rollout['observations'].size(0)).to(self.device)
        batch_info['episode_infos'] = rollout['episode_information']

    def off_policy_train_batch(self, batch_info: BatchInfo):
        """ Perform an 'off-policy' training step of sampling the replay buffer and gradient descent """
        rollout = self.env_roller.sample(batch_info, self.settings.number_of_steps, self.model)

        self.model.train()
        batch_info.optimizer.zero_grad()

        loss = self.policy_gradient.calculate_loss(
            batch_info=batch_info,
            device=self.device,
            model=self.model,
            rollout=rollout
        )

        loss.backward()

        # Gradient clipping
        if self.settings.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=self.settings.max_grad_norm
            )

            batch_info['gradient_norms'].append(grad_norm)

        batch_info.optimizer.step(closure=None)


class BufferedPolicyGradientReinforcerFactory(ReinforcerFactory):
    """ Factory class for the PolicyGradientReplayBuffer factory """
    def __init__(self, settings, env_factory: VecEnvFactory, model_factory: ModelFactory,
                 env_roller_factory: ReplayEnvRollerFactory, policy_gradient: PolicyGradientBase, parallel_envs: int, seed: int):
        self.settings = settings

        self.model_factory = model_factory
        self.env_factory = env_factory
        self.parallel_envs = parallel_envs
        self.env_roller_factory = env_roller_factory
        self.policy_gradient = policy_gradient
        self.seed = seed

    def instantiate(self, device: torch.device) -> ReinforcerBase:
        env = self.env_factory.instantiate(parallel_envs=self.parallel_envs, seed=self.seed)
        model = self.model_factory.instantiate(action_space=env.action_space)
        env_roller = self.env_roller_factory.instantiate(env, device, self.settings)

        return BufferedPolicyGradientReinforcer(device, self.settings, env, model, env_roller, self.policy_gradient)
