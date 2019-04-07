import os
import pathlib
import re
import torch


from vel.api import ModelConfig, EpochInfo, TrainingInfo, Model, Storage
from .strategy.checkpoint_strategy import CheckpointStrategy


class ClassicStorage(Storage):
    """ Model and metric persistence - classic implementation """

    def __init__(self, model_config: ModelConfig, checkpoint_strategy: CheckpointStrategy, backend, streaming=None):
        self.model_config = model_config
        self.backend = backend
        self.streaming = streaming if streaming is not None else []
        self.checkpoint_strategy = checkpoint_strategy

        self.cleaned = False

    def last_epoch_idx(self):
        """ Return last checkpointed epoch idx for given configuration. Returns 0 if no results have been stored """
        return self._persisted_last_epoch()

    def reset(self, configuration: dict) -> None:
        """
        Whenever there was anything stored in the database or not, purge previous state and start
        new training process from scratch.
        """
        self.clean(0)
        self.backend.store_config(configuration)

    def load(self, train_info: TrainingInfo) -> (dict, dict):
        """
        Resume learning process and return loaded hidden state dictionary
        """
        last_epoch = train_info.start_epoch_idx

        model_state = torch.load(self.checkpoint_filename(last_epoch))
        hidden_state = torch.load(self.checkpoint_hidden_filename(last_epoch))

        self.checkpoint_strategy.restore(hidden_state)
        train_info.restore(hidden_state)

        return model_state, hidden_state

    def get_metrics_frame(self):
        """ Get a frame of metrics from backend """
        return self.backend.get_frame()

    def clean(self, global_epoch_idx):
        """ Clean old checkpoints """
        if self.cleaned:
            return

        self.cleaned = True
        self.backend.clean(global_epoch_idx)

        self._make_sure_dir_exists()

        for x in os.listdir(self.model_config.checkpoint_dir()):
            match = re.match('checkpoint_(\\d+)\\.data', x)

            if match:
                idx = int(match[1])

                if idx > global_epoch_idx:
                    os.remove(os.path.join(self.model_config.checkpoint_dir(), x))

            match = re.match('checkpoint_hidden_(\\d+)\\.data', x)

            if match:
                idx = int(match[1])

                if idx > global_epoch_idx:
                    os.remove(os.path.join(self.model_config.checkpoint_dir(), x))

            match = re.match('checkpoint_best_(\\d+)\\.data', x)

            if match:
                idx = int(match[1])

                if idx > global_epoch_idx:
                    os.remove(os.path.join(self.model_config.checkpoint_dir(), x))

    def checkpoint(self, epoch_info: EpochInfo, model: Model):
        """ When epoch is done, we persist the training state """
        self.clean(epoch_info.global_epoch_idx - 1)

        self._make_sure_dir_exists()

        # Checkpoint latest
        torch.save(model.state_dict(), self.checkpoint_filename(epoch_info.global_epoch_idx))

        hidden_state = epoch_info.state_dict()
        self.checkpoint_strategy.write_state_dict(hidden_state)

        torch.save(hidden_state, self.checkpoint_hidden_filename(epoch_info.global_epoch_idx))

        if epoch_info.global_epoch_idx > 1 and self.checkpoint_strategy.should_delete_previous_checkpoint(
                                                   epoch_info.global_epoch_idx):
            prev_epoch_idx = epoch_info.global_epoch_idx - 1

            os.remove(self.checkpoint_filename(prev_epoch_idx))
            os.remove(self.checkpoint_hidden_filename(prev_epoch_idx))

        if self.checkpoint_strategy.should_store_best_checkpoint(epoch_info.global_epoch_idx, epoch_info.result):
            best_checkpoint_idx = self.checkpoint_strategy.current_best_checkpoint_idx

            if best_checkpoint_idx is not None:
                os.remove(self.checkpoint_best_filename(best_checkpoint_idx))

            torch.save(model.state_dict(), self.checkpoint_best_filename(epoch_info.global_epoch_idx))

            self.checkpoint_strategy.store_best_checkpoint_idx(epoch_info.global_epoch_idx)

        self.backend.store(epoch_info.result)

    def streaming_callbacks(self) -> list:
        """ Lift of callbacks for live streaming results """
        return self.streaming

    ####################################################################################################################
    # Filename helpers
    def checkpoint_filename(self, epoch_idx) -> str:
        """ Return checkpoint filename for this model """
        return self.model_config.checkpoint_dir('checkpoint_{:08}.data'.format(epoch_idx))

    def checkpoint_best_filename(self, epoch_idx) -> str:
        """ Return checkpoint filename for this model - best version """
        return self.model_config.checkpoint_dir('checkpoint_best_{:08}.data'.format(epoch_idx))

    def checkpoint_hidden_filename(self, epoch_idx) -> str:
        """ Return checkpoint filename for this model - hidden state """
        return self.model_config.checkpoint_dir('checkpoint_hidden_{:08}.data'.format(epoch_idx))

    ####################################################################################################################
    # Internal interface
    def _persisted_last_epoch(self) -> int:
        """ Return number of last epoch already calculated """
        epoch_number = 0
        self._make_sure_dir_exists()

        for x in os.listdir(self.model_config.checkpoint_dir()):
            match = re.match('checkpoint_(\\d+)\\.data', x)
            if match:
                idx = int(match[1])

                if idx > epoch_number:
                    epoch_number = idx

        return epoch_number

    def _make_sure_dir_exists(self):
        """ Make sure directory exists """
        filename = self.model_config.checkpoint_dir()
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)


def create(model_config, backend, checkpoint_strategy, streaming=None):
    """ Vel factory function """
    return ClassicStorage(
        model_config=model_config,
        backend=backend,
        checkpoint_strategy=checkpoint_strategy,
        streaming=streaming
    )
