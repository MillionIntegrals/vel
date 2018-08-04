import collections
import time
import typing
from dataclasses import dataclass

import numpy as np
import torch

from torch.optim import Optimizer

import waterboy.util.math as math_util

from waterboy.api.base import LinearBackboneModel, Model, ModelAugmentor
from waterboy.api.metrics import EpochResultAccumulator, BaseMetric
from waterboy.api.metrics.averaging_metric import AveragingMetric, AveragingNamedMetric
from waterboy.api.metrics.summing_metric import SummingNamedMetric
from waterboy.api.progress_idx import EpochIdx, BatchIdx
from waterboy.rl.api.base import ReinforcerBase, ReinforcerFactory, VecEnvFactoryBase
from waterboy.rl.env_roller.step_env_roller import StepEnvRoller


class PolicyGradientBase:
    """ Base class for policy gradient calculations """
    def calculate_loss(self, batch_idx, device, model, rollout, data_dict=None):
        """ Calculate loss of the supplied rollout """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


@dataclass
class PolicyGradientSettings:
    """ Settings for a policy gradient reinforcer"""
    policy_gradient: PolicyGradientBase
    vec_env: VecEnvFactoryBase
    model_augmentors: typing.List[ModelAugmentor]
    parallel_envs: int
    number_of_steps: int
    discount_factor: float
    seed: int
    max_grad_norm: float = None
    gae_lambda: float = 1.0
    batch_size: int = 256
    experience_replay: int = 1


class FPSMetric(AveragingMetric):
    """ Metric calculating FPS values """
    def __init__(self):
        super().__init__('fps')

        self.start_time = time.time()
        self.frames = 0

    def _value_function(self, data_dict):
        self.frames += data_dict['frames'].item()

        nseconds = time.time()-self.start_time
        fps = int(self.frames/nseconds)
        return fps


class EpisodeRewardMetric(BaseMetric):
    def __init__(self):
        super().__init__("episode_reward")
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, data_dict):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(data_dict['episode_infos'])

    def reset(self):
        """ Reset value of a metric """
        # Because it's a queue no need for reset..
        pass

    def value(self):
        """ Return current value for the metric """
        if self.buffer:
            return np.mean([ep['r'] for ep in self.buffer])
        else:
            return 0.0


class EpisodeLengthMetric(BaseMetric):
    def __init__(self):
        super().__init__("episode_length")
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, data_dict):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(data_dict['episode_infos'])

    def reset(self):
        """ Reset value of a metric """
        # Because it's a queue no need for reset..
        pass

    def value(self):
        """ Return current value for the metric """
        if self.buffer:
            return np.mean([ep['l'] for ep in self.buffer])
        else:
            return 0


class ExplainedVariance(AveragingMetric):
    """ How much value do rewards explain """
    def __init__(self):
        super().__init__("explained_variance")

    def _value_function(self, data_dict):
        values = data_dict['values']
        rewards = data_dict['rewards']

        explained_variance = 1 - torch.var(rewards - values) / torch.var(rewards)
        return explained_variance.item()


class PolicyGradientReinforcer(ReinforcerBase):
    """ Train network using a policy gradient algorithm """
    def __init__(self, device: torch.device, settings: PolicyGradientSettings, model: Model) -> None:
        self.device = device
        self.settings = settings

        self.environment = self.settings.vec_env.instantiate(
            parallel_envs=self.settings.parallel_envs, seed=self.settings.seed
        )

        self._internal_model = model

        augmentor_dict = {'env': self.environment}

        for augmentor in self.settings.model_augmentors:
            self._internal_model = augmentor.augment(self._internal_model, augmentor_dict)

        self._internal_model = self._internal_model.to(self.device)

        self.env_roller = StepEnvRoller(
            self.environment, self.device, self.settings.number_of_steps, self.settings.discount_factor,
            gae_lambda=self.settings.gae_lambda
        )

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            SummingNamedMetric("frames", reset_value=False),
            FPSMetric(),
            EpisodeRewardMetric(),
            EpisodeLengthMetric(),
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

    def train_step(self, batch_idx: BatchIdx, optimizer: Optimizer, result_accumulator: EpochResultAccumulator=None) -> None:
        """ Single, most atomic 'step' of learning this reinforcer can perform """
        # Calculate environment rollout on the evaluation version of the model
        self.model.eval()
        rollout = self.env_roller.rollout(self.model)

        rollout_tensors = {k: v for k, v in rollout.items() if isinstance(v, torch.Tensor)}
        rollout_size = next(v.size(0) for v in rollout_tensors.values())
        indices = np.arange(rollout_size)

        batch_splits = math_util.divide_ceiling(rollout_size, self.settings.batch_size)

        # Perform the training step
        self.model.train()

        data_dict_accumulator = []

        for i in range(self.settings.experience_replay):
            # Repeat the experience N times
            np.random.shuffle(indices)

            for sub_indices in np.array_split(indices, batch_splits):
                batch_rollout = {k: v[sub_indices] for k, v in rollout_tensors.items()}
                output_data_dict = {}

                optimizer.zero_grad()

                loss = self.settings.policy_gradient.calculate_loss(
                    batch_idx=batch_idx,
                    device=self.device,
                    model=self.model,
                    rollout=batch_rollout,
                    data_dict=output_data_dict
                )

                loss.backward()

                # Gradient clipping
                if self.settings.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        max_norm=self.settings.max_grad_norm
                    )

                    output_data_dict['grad_norm'] = torch.tensor(grad_norm).to(self.device)

                optimizer.step(closure=None)

                data_dict_accumulator.append(output_data_dict)

        data_dict = {
            'frames': torch.tensor(rollout_size).to(self.device),
            'episode_infos': rollout['episode_information'],
            'advantage_norm': torch.norm(rollout['advantages']),
            'values': rollout['values'],
            'rewards': rollout['discounted_rewards']
        }

        # Put in aggregated
        data_dict_keys = {y for x in data_dict_accumulator for y in x.keys()}

        for key in data_dict_keys:
            # Just average all the statistics from the loss function
            data_dict[key] = torch.mean(torch.stack([d[key] for d in data_dict_accumulator]))

        # Even with all the experience replay, we count the single rollout as single metrics entry
        if result_accumulator is not None:
            result_accumulator.calculate(data_dict)

    def train_epoch(self, epoch_idx: EpochIdx, batches_per_epoch: int, optimizer: Optimizer,
                    callbacks: list, result_accumulator: EpochResultAccumulator=None) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
        for callback in callbacks:
            callback.on_epoch_begin(epoch_idx)

        for batch_idx_number in range(batches_per_epoch):
            progress_idx = BatchIdx(epoch_idx, batch_idx_number, batches_per_epoch=batches_per_epoch, extra={
                'progress_meter': result_accumulator.intermediate_value('frames') / epoch_idx.extra['total_frames']
            })

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


class PolicyGradientReinforcerFactory(ReinforcerFactory):
    def __init__(self, vec_env, policy_gradient, model_augmentors,
                 number_of_steps, parallel_envs, discount_factor, gae_lambda,
                 max_grad_norm, seed, batch_size=256, experience_replay=1) -> None:
        self.settings = PolicyGradientSettings(
            policy_gradient=policy_gradient,
            vec_env=vec_env,
            model_augmentors=model_augmentors,
            parallel_envs=parallel_envs,
            number_of_steps=number_of_steps,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            seed=seed,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            experience_replay=experience_replay
        )

    def instantiate(self, device: torch.device, model: LinearBackboneModel) -> ReinforcerBase:
        return PolicyGradientReinforcer(device, self.settings, model)
