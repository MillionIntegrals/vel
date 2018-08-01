import collections
import time
import typing
from dataclasses import dataclass

import numpy as np
import torch

from torch.optim import Optimizer

from waterboy.api.base import LinearBackboneModel, Model, ModelAugmentor
from waterboy.api.metrics import EpochResultAccumulator, BaseMetric
from waterboy.api.metrics.averaging_metric import AveragingMetric, AveragingNamedMetric
from waterboy.api.metrics.summing_metric import SummingNamedMetric
from waterboy.api.progress_idx import EpochIdx, BatchIdx
from waterboy.rl.api.base import ReinforcerBase, ReinforcerFactory, VecEnvFactoryBase
from waterboy.rl.env_roller.step_env_roller import StepEnvRoller


class PolicyGradientBase:
    """ Base class for policy gradient calculations """
    def calculate_loss(self, device, model, rollout, data_dict=None):
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
    max_grad_norm: float


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
            self.environment, self.device, self.settings.number_of_steps, self.settings.discount_factor
        )

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        my_metrics = [
            SummingNamedMetric("frames", reset_value=False),
            FPSMetric(),
            EpisodeRewardMetric(),
            EpisodeLengthMetric(),
            AveragingNamedMetric("advantage_norm")
        ]

        if self.settings.max_grad_norm is not None:
            my_metrics.append(AveragingNamedMetric("grad_norm"))

        return my_metrics + self.settings.policy_gradient.metrics()

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        return self._internal_model

    def train_step(self, optimizer: Optimizer, result_accumulator: EpochResultAccumulator=None) -> None:
        """ Single, most atomic 'step' of learning this reinforcer can perform """
        # Calculate environment rollout on the evaluation version of the model
        self.model.eval()
        rollout = self.env_roller.rollout(self.model)

        # Perform the training step
        self.model.train()
        optimizer.zero_grad()

        data_dict = {
            'frames': torch.tensor(rollout['observations'].shape[0]).to(self.device),
            'episode_infos': rollout['episode_information'],
            'advantage_norm': torch.norm(rollout['advantages'])
        }

        loss = self.settings.policy_gradient.calculate_loss(self.device, self.model, rollout, data_dict)
        loss.backward()

        # Gradient clipping
        if self.settings.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=self.settings.max_grad_norm
            )

            data_dict['grad_norm'] = torch.tensor(grad_norm).to(self.device)

        optimizer.step(closure=None)

        if result_accumulator is not None:
            result_accumulator.calculate(data_dict)

    def train_epoch(self, epoch_idx: EpochIdx, batches_per_epoch: int, optimizer: Optimizer,
                    callbacks: list, result_accumulator: EpochResultAccumulator=None) -> None:
        """ Train model on an epoch of a fixed number of batch updates """
        for callback in callbacks:
            callback.on_epoch_begin(epoch_idx)

        for batch_idx_number in range(batches_per_epoch):
            progress_idx = BatchIdx(epoch_idx, batch_idx_number, batches_per_epoch=batches_per_epoch)

            for callback in callbacks:
                callback.on_batch_begin(progress_idx)

            self.train_step(optimizer, result_accumulator)

            for callback in callbacks:
                callback.on_batch_end(progress_idx, result_accumulator.value(), optimizer)

        result_accumulator.freeze_results()

        epoch_result = result_accumulator.result()

        for callback in callbacks:
            callback.on_epoch_end(epoch_idx, epoch_result)

        return epoch_result


class PolicyGradientReinforcerFactory(ReinforcerFactory):
    def __init__(self, vec_env, policy_gradient, model_augmentors,
                 number_of_steps, parallel_envs, discount_factor,
                 max_grad_norm, seed) -> None:
        self.settings = PolicyGradientSettings(
            policy_gradient=policy_gradient,
            vec_env=vec_env,
            model_augmentors=model_augmentors,
            parallel_envs=parallel_envs,
            number_of_steps=number_of_steps,
            discount_factor=discount_factor,
            seed=seed,
            max_grad_norm=max_grad_norm
        )

    def instantiate(self, device: torch.device, model: LinearBackboneModel) -> ReinforcerBase:
        return PolicyGradientReinforcer(device, self.settings, model)
