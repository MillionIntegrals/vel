"""
Learning rate finder.
Loosely based on: https://github.com/fastai/fastai/blob/master/fastai/learner.py
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import waterboy.util.intepolate as interp

from waterboy.api import Learner
from waterboy.api.metrics import EpochResultAccumulator, TrainingHistory


class LrFindCommand:
    """Helps you find an optimal learning rate for a model.

     It uses the technique developed in the 2015 paper
     `Cyclical Learning Rates for Training Neural Networks`, where
     we simply keep increasing the learning rate from a very small value,
     until the loss starts decreasing.

    Args:
        start_lr (float/numpy array) : Passing in a numpy array allows you
            to specify learning rates for a learner's layer_groups
        end_lr (float) : The maximum learning rate to try.
        wds (iterable/float)

    Examples:
        As training moves us closer to the optimal weights for a model,
        the optimal learning rate will be smaller. We can take advantage of
        that knowledge and provide lr_find() with a starting learning rate
        1000x smaller than the model's current learning rate as such:

        >> learn.lr_find(lr/1000)

        >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
        >> learn.lr_find(lrs / 1000)

    Notes:
        lr_find() may finish before going through each batch of examples if
        the loss decreases enough.

    .. _Cyclical Learning Rates for Training Neural Networks:
        http://arxiv.org/abs/1506.01186

    """
    def __init__(self, model_config, model, source, optimizer, start_lr=1e-5, end_lr=10, num_it=100,
                 interpolation='logscale', freeze=False, stop_dv=True, divergence_threshold=4.0, metric='train:loss'):
        # Mandatory pieces
        self.model = model
        self.source = source
        self.optimizer = optimizer
        self.model_config = model_config
        # Settings
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.interpolation = interpolation
        self.freeze = freeze
        self.stop_dv = stop_dv
        self.divergence_threshold = divergence_threshold
        self.metric = metric

    def run(self):
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model)

        lr_schedule = interp.interpolate_series(self.start_lr, self.end_lr, self.num_it, self.interpolation)

        iterator = iter(self.source.train_loader)

        metrics = learner.metrics()
        history = TrainingHistory()

        if self.freeze:
            learner.model.freeze()

        # Optimizer shoudl be created after freeze
        optimizer_instance = self.optimizer(filter(lambda p: p.requires_grad, learner.model.parameters()))

        learner.train()

        best_value = None

        result_accumulator = EpochResultAccumulator(1, metrics)

        for iteration_idx, lr in enumerate(tqdm.tqdm(lr_schedule)):
            # First, set the learning rate, the same for each parameter group
            for param_group in optimizer_instance.param_groups:
                param_group['lr'] = lr

            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(self.source.train_loader)
                data, target = next(iterator)

            learner.train_batch(data, target, optimizer_instance, result_accumulator=result_accumulator)

            # METRIC RECORDING PART
            current_value = result_accumulator.intermediate_value(self.metric)

            final_metrics = {'epoch_idx': iteration_idx, self.metric: current_value, 'lr': lr}

            if best_value is None or current_value < best_value:
                best_value = current_value

            # Stop on divergence
            if self.stop_dv and (np.isnan(current_value) or current_value > best_value * self.divergence_threshold):
                break

            history.add(final_metrics)

        frame = history.frame()

        fig, ax = plt.subplots(1, 2)

        # ax[0].set_title('Learning rate')
        ax[0].plot(frame.index, frame.lr)
        ax[0].set_title("LR Schedule")
        ax[0].set_xlabel("Num iterations")
        ax[0].set_ylabel("Learning rate")

        if self.interpolation == 'logscale':
            ax[0].set_yscale("log", nonposy='clip')

        ax[1].plot(frame.lr, frame[self.metric], label=self.metric)
        # ax[1].plot(frame.lr, frame[self.metric].ewm(com=20).mean(), label=self.metric + ' smooth')
        ax[1].set_title(self.metric)
        ax[1].set_xlabel("Learning rate")
        ax[1].set_ylabel(self.metric)
        # ax[1].legend()

        if self.interpolation == 'logscale':
            ax[1].set_xscale("log", nonposx='clip')

        plt.show()


def create(model_config, model, source, optimizer, start_lr=1e-5, end_lr=10, iterations=100, freeze=False,
           interpolation='logscale', stop_dv=True, divergence_threshold=4.0, metric='train:loss'):
    """ Create a learning rate finder """
    return LrFindCommand(
        model_config=model_config,
        model=model,
        source=source,
        optimizer=optimizer,
        start_lr=start_lr,
        end_lr=end_lr,
        num_it=iterations,
        interpolation=interpolation,
        freeze=freeze,
        stop_dv=stop_dv,
        divergence_threshold=divergence_threshold,
        metric=metric
    )
