import vel.api as base

from vel.api import BatchInfo, TrainingInfo


class LinearBatchScaler(base.Callback):
    """ Scales linearly LR from maximum value to 0 through every batch of training """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.starting_lr = None

    def on_initialization(self, training_info: TrainingInfo):
        self.starting_lr = [p['lr'] for p in self.optimizer.param_groups]

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['linear_batch_scaler/starting_lr'] = self.starting_lr

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        self.starting_lr = hidden_state_dict['linear_batch_scaler/starting_lr']

    def on_batch_begin(self, batch_info: BatchInfo):
        for starting_lr, param_group in zip(self.starting_lr, self.optimizer.param_groups):
            param_group['lr'] = starting_lr * (1.0 - batch_info['progress'])


class LinearBatchScalerFactory(base.SchedulerFactory):
    """ Factory class for linear batch scaler scheduler """
    def instantiate(self, optimizer, last_epoch=-1) -> LinearBatchScaler:
        return LinearBatchScaler(optimizer)


def create():
    """ Vel factory function """
    return LinearBatchScalerFactory()


