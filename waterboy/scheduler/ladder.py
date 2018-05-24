import torch.optim.lr_scheduler as scheduler
import numpy as np


class LadderScheduler:
    def __init__(self, optimizer, ladder, last_epoch):
        self.schedule_limits = np.cumsum([x[0] for x in ladder])
        self.schedule_numbers = np.array([float(x[1]) for x in ladder])
        self.scheduler = scheduler.LambdaLR(optimizer, self.lambda_fn, last_epoch=last_epoch)

    def get_lr(self):
        return self.scheduler.get_lr()

    def lambda_fn(self, epoch_idx):
        idx = np.minimum(np.searchsorted(self.schedule_limits, epoch_idx), len(self.schedule_limits) - 1)
        return self.schedule_numbers[idx]

    def pre_epoch_step(self, epoch_idx):
        self.scheduler.step(epoch=epoch_idx)

    def post_epoch_step(self, metrics):
        pass


def create(ladder):
    """ Create a hand-scheduled ladder scheduler """
    def scheduler_fn(optimizer, last_epoch=-1):
        return LadderScheduler(optimizer, ladder, last_epoch)

    return scheduler_fn
