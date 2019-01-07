import time

from vel.api import BatchInfo, TrainingInfo, Callback


class TimeTracker(Callback):
    """ Track training time - in seconds """
    def __init__(self):
        self.start_time = None

    def on_initialization(self, training_info: TrainingInfo):
        training_info['time'] = 0.0

    def on_train_begin(self, training_info: TrainingInfo):
        self.start_time = time.time()

    def on_batch_end(self, batch_info: BatchInfo):
        current_time = time.time()
        batch_time = current_time - self.start_time
        self.start_time = current_time

        batch_info['time'] = batch_time
        batch_info.training_info['time'] += batch_info['time']

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['time_tracker/time'] = training_info['time']

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        training_info['time'] = hidden_state_dict['time_tracker/time']
