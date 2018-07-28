import torch.optim.lr_scheduler as scheduler


# class ReduceLrOnPlateau:
#     def __init__(self, optimizer, metric_name, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, epsilon):
#         self.metric_name = metric_name
#         self.scheduler = scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode=mode,
#             factor=factor,
#             patience=patience,
#             threshold=threshold,
#             threshold_mode=threshold_mode,
#             cooldown=cooldown,
#             min_lr=min_lr,
#             eps=epsilon
#         )
#
#     def get_lr(self):
#         return [p['lr'] for p in self.scheduler.optimizer.param_groups]
#
#     def pre_epoch_step(self):
#         pass
#
#     def post_epoch_step(self, metrics):
#         self.scheduler.step(metrics[self.metric_name])
#
#
# def create(metric_name, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=10,
#            min_lr=0, epsilon=1e-8):
#     """ Create a scheduler that lowers the LR on metric plateau """
#     def scheduler_fn(optimizer):
#         return ReduceLrOnPlateau(optimizer, metric_name, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, epsilon)
#
#     return scheduler_fn

