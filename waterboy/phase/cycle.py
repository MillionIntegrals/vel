import numpy as np

import waterboy.api.base as base
import waterboy.util.intepolate as interp
import waterboy.util.module_util as mu


class CycleCallback(base.Callback):
    """ A callback that manages setting the proper learning rate """

    def __init__(self, optimizer, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, init_iter=0, init_lr=0, interpolate='linear'):
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

    def on_batch_begin(self, progress_idx):
        """ Set proper learning rate """
        cycle_length = self.cycle_lengths[progress_idx.local_epoch_number - 1]
        cycle_start = self.cycle_starts[progress_idx.local_epoch_number - 1]

        numerator = (progress_idx.local_epoch_number - cycle_start - 1) * progress_idx.batches_per_epoch + progress_idx.batch_number
        denominator = cycle_length * progress_idx.batches_per_epoch

        interpolation_number = numerator / denominator

        if cycle_start == 0 and numerator < self.init_iter:
            lr = self.init_lr
        else:
            if isinstance(self.max_lr, list):
                lr = [interp.interpolate_single(max_lr, min_lr, interpolation_number, how=self.interpolate) for max_lr, min_lr in zip(self.max_lr, self.min_lr)]
            else:
                lr = interp.interpolate_single(self.max_lr, self.min_lr, interpolation_number, how=self.interpolate)

        self.set_lr(lr)

    def set_lr(self, lr):
        """ Set a learning rate for the optimizer """
        if isinstance(lr, list):
            for group_lr, param_group in zip(lr, self.optimizer.param_groups):
                param_group['lr'] = group_lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class CyclePhase(base.TrainPhase):
    """ Most generic phase of training """

    def __init__(self, optimizer_fn, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, interpolate='linear', init_lr=0, init_iter=0, freeze=False):
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

        self.optimizer_fn = optimizer_fn
        self.freeze = freeze

        self._optimizer_instance = None
        self._source = None
        self._metrics = []
        self._callbacks = []

        self.special_callback = None

    @property
    def number_of_epochs(self) -> int:
        return self.epochs

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Prepare the phase for learning """
        parameter_groups = mu.to_parameter_groups(learner.model.get_layer_groups())
        self._optimizer_instance = self.optimizer_fn(parameter_groups)
        self._source = source

        if metrics is not None:
            self._metrics = metrics

        self.special_callback = CycleCallback(
            self._optimizer_instance,
            max_lr=self.max_lr, min_lr=self.min_lr, cycles=self.cycles,
            cycle_len=self.cycle_len, cycle_mult=self.cycle_mult, interpolate=self.interpolate,
            init_iter=self.init_iter, init_lr=self.init_lr
        )

        if callbacks is not None:
            self._callbacks = [self.special_callback] + callbacks
        else:
            self._callbacks = [self.special_callback]

    def execute_epoch(self, epoch_idx, learner):
        """ Prepare the phase for learning """
        epoch_result = learner.run_epoch(
            epoch_idx, self._metrics, self._source, self._optimizer_instance, self._callbacks
        )

        return epoch_result

    def tear_down_phase(self, learner):
        """ Clean up after phase is done """
        pass


def create(optimizer, max_lr, min_lr, cycles, cycle_len=1, cycle_mult=1, interpolate='linear', init_lr=0, init_iter=0):
    """ Waterboy creation function """
    return CyclePhase(
        max_lr=max_lr,
        min_lr=min_lr,
        cycles=cycles,
        cycle_len=cycle_len,
        cycle_mult=cycle_mult,
        interpolate=interpolate,
        optimizer_fn=optimizer,
        init_lr=init_lr,
        init_iter=init_iter,
    )
