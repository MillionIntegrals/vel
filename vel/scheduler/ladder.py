import torch.optim.lr_scheduler as scheduler
import numpy as np


from vel.api import Callback, SchedulerFactory


class LadderScheduler(Callback):
    """ Scheduler defined by a set of learning rates after reaching given number of iterations """
    def __init__(self, optimizer, ladder, last_epoch):
        self.schedule_limits = np.cumsum([x[0] for x in ladder])
        self.schedule_numbers = np.array([float(x[1]) for x in ladder])
        self.scheduler = scheduler.LambdaLR(optimizer, self.lambda_fn, last_epoch=last_epoch)

    def lambda_fn(self, epoch_idx):
        idx = np.minimum(np.searchsorted(self.schedule_limits, epoch_idx), len(self.schedule_limits) - 1)
        return self.schedule_numbers[idx]

    def on_epoch_begin(self, epoch_info):
        self.scheduler.step(epoch=epoch_info.global_epoch_idx)


class LadderSchedulerFactory(SchedulerFactory):
    """ Factory class for ladder scheduler """
    def __init__(self, ladder):
        self.ladder = ladder

    def instantiate(self, optimizer, last_epoch=-1) -> LadderScheduler:
        return LadderScheduler(optimizer, self.ladder, last_epoch)


def create(ladder):
    """ Vel factory function """
    return LadderSchedulerFactory(ladder)
