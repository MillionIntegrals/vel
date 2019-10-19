import typing
import collections

from vel.api import BatchInfo, TrainingInfo, Callback


class SampleTracker(Callback):
    """ Callback that calculates number of samples processed during the training process """

    def on_initialization(self, training_info: TrainingInfo):
        training_info['samples'] = collections.defaultdict(int)

    def on_batch_end(self, batch_info: BatchInfo, dataset: typing.Optional[str] = None) -> None:
        samples = batch_info['datapoint']['x'].shape[0]

        batch_info['samples'] = samples

        if dataset is not None:
            batch_info.training_info['samples'][dataset] += samples

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['sample_tracker/samples'] = training_info['samples']

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        training_info['samples'] = hidden_state_dict['sample_tracker/samples']
