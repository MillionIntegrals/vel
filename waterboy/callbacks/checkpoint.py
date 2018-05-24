import os.path
import torch
import pathlib
import re

import waterboy.api.callback as cb


class Checkpoint(cb.Callback):
    """ Save model weights to the file """
    def __init__(self, model_config, checkpoint_frequency=None, metric='val:loss', metric_mode='min', initial_value=None, initial_epoch=None):
        self.model_config = model_config

        self.checkpoint_frequency = checkpoint_frequency

        # For checkpointing the best
        self.metric = metric
        self.metric_mode = metric_mode

        self.best_value = initial_value
        self.best_epoch = initial_epoch

    def last_epoch(self):
        """ Return number of last epoch already calculated """
        epoch_number = 0

        for x in os.listdir(self.model_config.checkpoint_dir()):
            match = re.match('checkpoint_(\\d+)\\.npy', x)
            if match:
                idx = int(match[1])

                if idx > epoch_number:
                    epoch_number = idx

        return epoch_number

    def load_model(self, epoch_idx, model, optimizer):
        """ Load model and optimizer state from state file """
        model.load_state_dict(torch.load(self.model_config.checkpoint_filename(epoch_idx)))
        optimizer.load_state_dict(torch.load(self.model_config.checkpoint_opt_filename(epoch_idx)))

    def _is_better(self, old_value, new_value):
        """ Check if new metric is better than the old"""
        if self.metric_mode == 'min' and (new_value < old_value):
            return True
        elif self.metric_mode == 'max' and (new_value > old_value):
            return True
        else:
            return False

    def on_epoch_end(self, epoch_idx, metrics, model, optimizer):
        filename = self.model_config.checkpoint_filename(epoch_idx)
        pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)

        # Checkpoint latest
        torch.save(model.state_dict(), self.model_config.checkpoint_filename(epoch_idx))
        torch.save(optimizer.state_dict(), self.model_config.checkpoint_opt_filename(epoch_idx))

        if epoch_idx > 1:
            prev_epoch_idx = epoch_idx - 1
            # Maybe we need to delete a previous checkpoint

            if self.checkpoint_frequency is not None and prev_epoch_idx % self.checkpoint_frequency == 0:
                # It was a checkpoint frequency
                return

            os.remove(self.model_config.checkpoint_filename(prev_epoch_idx))
            os.remove(self.model_config.checkpoint_opt_filename(prev_epoch_idx))

        if self.metric is not None:
            metric_value = metrics[self.metric]

            if self.best_value is None:
                self.best_value = metric_value
                self.best_epoch = epoch_idx
                torch.save(model.state_dict(), self.model_config.checkpoint_best_filename(self.best_epoch))
            elif self._is_better(self.best_value, metric_value):
                if os.path.exists(self.model_config.checkpoint_best_filename(self.best_epoch)):
                    os.remove(self.model_config.checkpoint_best_filename(self.best_epoch))

                self.best_value = metric_value
                self.best_epoch = epoch_idx

                torch.save(model.state_dict(), self.model_config.checkpoint_best_filename(self.best_epoch))


def create(model_config):
    """ Create a checpoint callback """
    return Checkpoint(model_config)
