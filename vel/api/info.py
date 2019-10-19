import collections.abc as abc
import numpy as np
import pandas as pd
import typing

import torch

from vel.exception import VelException


class TrainingHistory:
    """
    Simple aggregator for the training history.

    An output of training storing scalar metrics in a pandas dataframe.
    """
    def __init__(self):
        self.data = []

    def add(self, epoch_result):
        """ Add a datapoint to the history """
        self.data.append(epoch_result)

    def frame(self):
        """ Return history dataframe """
        return pd.DataFrame(self.data).set_index('epoch_idx')


class TrainingInfo(abc.MutableMapping):
    """
    Information that need to persist through the whole training process

    Data dict is any extra information processes may want to store
    """

    def __init__(self, start_epoch_idx=0, metrics=None, callbacks=None):
        self.data_dict = {}

        self.start_epoch_idx = start_epoch_idx
        self.metrics = metrics if metrics is not None else []
        self.callbacks = callbacks if callbacks is not None else []
        self.history = TrainingHistory()

    def restore(self, hidden_state):
        """ Restore any state from checkpoint - currently not implemented but possible to do so in the future """
        for callback in self.callbacks:
            callback.load_state_dict(self, hidden_state)

    def initialize(self):
        """
        Runs for the first time a training process is started from scratch. Is guaranteed to be run only once
        for the training process. Will be run before all other callbacks.
        """
        for callback in self.callbacks:
            callback.on_initialization(self)

    def on_train_begin(self):
        """
        Beginning of a training process - is run every time a training process is started, even if it's restarted from
        a checkpoint.
        """
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def on_train_end(self):
        """
        Finalize training process. Runs each time at the end of a training process.
        """
        for callback in self.callbacks:
            callback.on_train_end(self)

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __contains__(self, item):
        return item in self.data_dict


class EpochResultAccumulator:
    """ Result of a single epoch of training -- accumulator """
    def __init__(self, global_epoch_idx, metrics):
        self.global_epoch_idx = global_epoch_idx
        self.metrics = metrics

        self.frozen_results = {}

        self._reset_metrics()
        self.metrics_by_name = {m.name: m for m in self.metrics}

    def __contains__(self, metric):
        if ':' in metric:
            # TODO(jerry) There's got to be a better way to do it
            metric_name = metric.split(':')[-1]
        else:
            metric_name = metric

        return metric_name in self.metrics_by_name

    @torch.no_grad()
    def calculate(self, batch_info):
        """ Calculate metric values """
        for m in self.metrics:
            m.calculate(batch_info)

    def _reset_metrics(self):
        """ Internal API : reset state of metrics """
        for m in self.metrics:
            m.reset()

    def value(self, dataset=None):
        """ Return current dictionary value of the metrics """
        from vel.metric import MetricKey
        return {MetricKey(dataset, m.name, m.scope): m.value() for m in self.metrics}

    def intermediate_value(self, metric):
        """ Return an intermediate (inter-epoch) value of a metric """
        if ':' in metric:
            # TODO(jerry) There's got to be a better way to do it
            metric_name = metric.split(':')[-1]
        else:
            metric_name = metric

        return self.metrics_by_name[metric_name].value()

    def freeze_results(self, dataset=None):
        self.frozen_results.update(self.value(dataset))
        self._reset_metrics()

    def result(self):
        """ Return the epoch result """
        final_result = {}

        for key, value in self.frozen_results.items():
            final_result[key] = value

        return final_result


class EpochInfo(abc.MutableMapping):
    """
    Information that need to persist through the single epoch.

    Global epoch index - number of epoch from start of training until now
    Local epoch index - number of epoch from start of current "phase" until now

    Data dict is any extra information processes may want to store
    """

    def __init__(self, training_info: TrainingInfo, global_epoch_idx: int, batches_per_epoch: int,
                 optimizer: typing.Optional[torch.optim.Optimizer] = None, local_epoch_idx: int = None,
                 callbacks: typing.Optional[list] = None):
        self.training_info = training_info
        self.optimizer = optimizer
        self.batches_per_epoch = batches_per_epoch

        self.global_epoch_idx = global_epoch_idx

        if local_epoch_idx is None:
            self.local_epoch_idx = self.global_epoch_idx
        else:
            self.local_epoch_idx = local_epoch_idx

        self.result_accumulator = EpochResultAccumulator(self.global_epoch_idx, self.training_info.metrics)
        self._result = {}
        self.data_dict = {}

        if callbacks is None:
            self.callbacks = self.training_info.callbacks
        else:
            self.callbacks = callbacks

    def state_dict(self) -> dict:
        """ Calculate hidden state dictionary """
        hidden_state = {}

        if self.optimizer is not None:
            hidden_state['optimizer'] = self.optimizer.state_dict()

        for callback in self.callbacks:
            callback.write_state_dict(self.training_info, hidden_state)

        return hidden_state

    def on_epoch_begin(self) -> None:
        """ Initialize an epoch """
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self):
        """ Finish epoch processing """
        self.freeze_epoch_result()

        for callback in self.callbacks:
            callback.on_epoch_end(self)

        self.training_info.history.add(self.result)

    @property
    def metrics(self):
        """ Just forward metrics from training_info """
        return self.training_info.metrics

    def freeze_epoch_result(self):
        """ Calculate epoch 'metrics' result and store it in the internal variable """
        self._result = self.result_accumulator.result()

    @property
    def result(self) -> dict:
        """ Result of the epoch """
        if self._result is None:
            raise VelException("Result has not been frozen yet")
        else:
            return self._result

    def __repr__(self):
        return f"EpochInfo(global_epoch_idx={self.global_epoch_idx}, local_epoch_idx={self.local_epoch_idx})"

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __contains__(self, item):
        return item in self.data_dict


class BatchInfo(abc.MutableMapping):
    """
    Information about current batch.

    Serves as a dictionary where all the information from the single run are accumulated
    """
    def __init__(self, epoch_info: EpochInfo, batch_number: int):
        self.epoch_info = epoch_info
        self.batch_number = batch_number
        self.data_dict = {}

    def on_batch_begin(self, dataset=None):
        """ Initialize batch processing """
        for callback in self.callbacks:
            callback.on_batch_begin(self, dataset)

    def on_batch_end(self, dataset=None):
        """ Finalize batch processing """
        for callback in self.callbacks:
            callback.on_batch_end(self, dataset)

        # Even with all the experience replay, we count the single rollout as a single batch
        self.epoch_info.result_accumulator.calculate(self)

    @property
    def aggregate_batch_number(self):
        return self.batch_number + self.epoch_info.batches_per_epoch * (self.epoch_info.global_epoch_idx - 1)

    @property
    def epoch_number(self):
        return self.epoch_info.global_epoch_idx

    @property
    def batches_per_epoch(self):
        return self.epoch_info.batches_per_epoch

    @property
    def local_epoch_number(self):
        return self.epoch_info.local_epoch_idx

    @property
    def optimizer(self):
        return self.epoch_info.optimizer

    @property
    def training_info(self):
        return self.epoch_info.training_info

    @property
    def callbacks(self):
        return self.epoch_info.callbacks

    def aggregate_key(self, aggregate_key):
        """ Aggregate values from key and put them into the top-level dictionary """
        aggregation = self.data_dict[aggregate_key]  # List of dictionaries of numpy arrays/scalars

        # Aggregate sub batch data
        data_dict_keys = {y for x in aggregation for y in x.keys()}

        for key in data_dict_keys:
            # Just average all the statistics from the loss function
            stacked = np.stack([d[key] for d in aggregation], axis=0)
            self.data_dict[key] = np.mean(stacked, axis=0)

    def drop_key(self, key):
        """ Remove key from dictionary """
        del self.data_dict[key]

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __contains__(self, item):
        return item in self.data_dict

    def __repr__(self):
        return f"[BatchInfo epoch:{self.epoch_info.global_epoch_idx} batch:{self.batch_number}/{self.batches_per_epoch}]"  # noqa
