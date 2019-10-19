import torch.optim.lr_scheduler as scheduler

from vel.api import Callback, SchedulerFactory, EpochInfo


class MultiStepScheduler(Callback):
    """ Scheduler multiplying the learning rate given number after given number of epochs """
    def __init__(self, optimizer, milestones, gamma, last_epoch):
        self.scheduler = scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return self.scheduler.get_lr()

    def on_epoch_end(self, epoch_info: EpochInfo) -> None:
        self.scheduler.step(epoch=epoch_info.global_epoch_idx)


class MultiStepSchedulerFactory(SchedulerFactory):
    """ Factory class for ladder scheduler """
    def __init__(self, milestones, gamma):
        self.milestones = milestones
        self.gamma = gamma

    def instantiate(self, optimizer, last_epoch=-1) -> MultiStepScheduler:
        return MultiStepScheduler(optimizer, self.milestones, self.gamma, last_epoch)


def create(milestones, gamma=0.1):
    """ Create a multi-step scheduler """
    return MultiStepSchedulerFactory(milestones, gamma)
