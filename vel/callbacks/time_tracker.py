import time

from vel.api import BatchInfo, TrainingInfo
from vel.api.base import Callback


class TimeTracker(Callback):
    """ Track training time - in seconds """
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, training_info: TrainingInfo):
        self.start_time = time.time()
        training_info['time'] = 0.0

    def on_batch_end(self, batch_info: BatchInfo):
        current_time = time.time()
        batch_time = current_time - self.start_time
        self.start_time = current_time

        batch_info['time'] = batch_time
        batch_info.training_info['time'] += batch_info['time']
