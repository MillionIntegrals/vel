import torch.optim.lr_scheduler as scheduler


# class MultiStepScheduler:
#     def __init__(self, optimizer, milestones, gamma, last_epoch):
#         self.scheduler = scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch=last_epoch)
#
#     def get_lr(self):
#         return self.scheduler.get_lr()
#
#     def pre_epoch_step(self, epoch_idx):
#         self.scheduler.step()
#
#     def post_epoch_step(self, epoch_idx, metrics):
#         pass
#
#
# def create(milestones, gamma=0.1):
#     """ Create a multi-step scheduler """
#     def scheduler_fn(optimizer, last_epoch=-1):
#         return MultiStepScheduler(optimizer, milestones, gamma, last_epoch=last_epoch)
#
#     return scheduler_fn
