import typing
import numpy as np

import vel.util.interpolate as interp

from vel.api import BatchInfo, EpochInfo, TrainingInfo, Callback, OptimizedModel
from vel.train import TrainPhase


class CycleCallback(Callback):
    """ A callback that manages setting the proper learning rate """

    def __init__(self, optimizer, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, init_iter=0, init_lr=0,
                 interpolate='linear'):
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cycles = cycles
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult

        self.init_iter = init_iter
        self.init_lr = init_lr

        if cycle_mult > 1:
            self.epochs = self.cycle_len * (self.cycle_mult ** self.cycles - 1) // (self.cycle_mult - 1)
        else:
            self.epochs = self.cycle_mult * self.cycles

        self.optimizer = optimizer
        self.interpolate = interpolate

        # self.current_cycle = None
        self.cycle_dict, self.cycle_lengths, self.cycle_starts = self._init_cycle_dict()

    def _init_cycle_dict(self):
        """ Populate a cycle dict """
        dict_arr = np.zeros(self.epochs, dtype=int)
        length_arr = np.zeros(self.epochs, dtype=int)
        start_arr = np.zeros(self.epochs, dtype=int)

        c_len = self.cycle_len
        idx = 0

        for i in range(self.cycles):
            current_start = idx
            for j in range(c_len):
                dict_arr[idx] = i
                length_arr[idx] = c_len
                start_arr[idx] = current_start
                idx += 1

            c_len *= self.cycle_mult

        return dict_arr, length_arr, start_arr

    def on_batch_begin(self, batch_info: BatchInfo, dataset: typing.Optional[str] = None):
        """ Set proper learning rate """
        cycle_length = self.cycle_lengths[batch_info.local_epoch_number - 1]
        cycle_start = self.cycle_starts[batch_info.local_epoch_number - 1]

        numerator = (
            (batch_info.local_epoch_number - cycle_start - 1) * batch_info.batches_per_epoch + batch_info.batch_number
        )
        denominator = cycle_length * batch_info.batches_per_epoch

        interpolation_number = numerator / denominator

        if cycle_start == 0 and numerator < self.init_iter:
            lr = self.init_lr
        else:
            if isinstance(self.max_lr, list):
                lr = [
                    interp.interpolate_single(max_lr, min_lr, interpolation_number, how=self.interpolate)
                    for max_lr, min_lr in zip(self.max_lr, self.min_lr)
                ]
            else:
                lr = interp.interpolate_single(self.max_lr, self.min_lr, interpolation_number, how=self.interpolate)

        self.optimizer.set_lr(lr)


class CyclePhase(TrainPhase):
    """ Most generic phase of training """

    def __init__(self, optimizer_factory, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, interpolate='linear',
                 init_lr=0, init_iter=0, freeze=False):
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.cycles = cycles
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult

        if cycle_mult > 1:
            self.epochs = self.cycle_len * (self.cycle_mult ** self.cycles - 1) // (self.cycle_mult - 1)
        else:
            self.epochs = self.cycle_mult * self.cycles

        self.interpolate = interpolate

        self.init_iter = init_iter
        self.init_lr = init_lr

        self.optimizer_factory = optimizer_factory
        self.freeze = freeze

        self._optimizer_instance = None
        self._loader = None

        self.special_callback = None

    @property
    def number_of_epochs(self) -> int:
        return self.epochs

    def set_up_phase(self, training_info, model: OptimizedModel, loader):
        """ Prepare the phase for learning """
        # To parameter groups handles properly filtering parameters that don't require gradient
        self._optimizer_instance = model.create_optimizer(self.optimizer_factory)
        self._loader = loader

        self.special_callback = CycleCallback(
            self._optimizer_instance,
            max_lr=self.max_lr, min_lr=self.min_lr, cycles=self.cycles,
            cycle_len=self.cycle_len, cycle_mult=self.cycle_mult, interpolate=self.interpolate,
            init_iter=self.init_iter, init_lr=self.init_lr
        )

        return self._optimizer_instance

    def epoch_info(self, training_info: TrainingInfo, global_idx: int, local_idx: int) -> EpochInfo:
        """ Create Epoch info """
        return EpochInfo(
            training_info=training_info,
            global_epoch_idx=global_idx,
            local_epoch_idx=local_idx,
            batches_per_epoch=self._loader.size['train'],
            optimizer=self._optimizer_instance,
            # Add special callback for this epoch
            callbacks=[self.special_callback] + training_info.callbacks
        )

    def execute_epoch(self, epoch_info, learner):
        """ Prepare the phase for learning """
        learner.run_epoch(epoch_info, self._loader)


def create(optimizer, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, interpolate='linear', init_lr=0, init_iter=0):
    """ Vel factory function """
    return CyclePhase(
        max_lr=max_lr,
        min_lr=min_lr,
        cycles=cycles,
        cycle_len=cycle_len,
        cycle_mult=cycle_mult,
        interpolate=interpolate,
        optimizer_factory=optimizer,
        init_lr=init_lr,
        init_iter=init_iter,
    )
