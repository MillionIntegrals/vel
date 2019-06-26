import sys
import torch
import torch.nn
import tqdm
import typing

from vel.api import GradientModel, TrainingInfo, EpochInfo, BatchInfo
from vel.data import DatasetLoader

from vel.util.tensor_util import to_device


class Trainer:
    """ Manages training process of a single model """

    def __init__(self, device: torch.device, model: GradientModel, max_grad_norm: typing.Optional[float] = None):
        self.device = device
        self.model = model.to(device)
        self.max_grad_norm = max_grad_norm

    def metrics(self):
        """ Return metrics for given learner/model """
        return self.model.metrics()

    def summary(self):
        """ Print summary for given learner/model """
        return self.model.summary()

    def train(self):
        """ Set model in the training mode """
        return self.model.train()

    def eval(self):
        """ Set model in the evaluation mode """
        return self.model.eval()

    def number_of_parameters(self):
        """ Count model parameters """
        return sum(p.numel() for p in self.model.parameters())

    def initialize_training(self, training_info: TrainingInfo, model_state=None, hidden_state=None):
        """ Prepare for training """
        if model_state is None:
            self.model.reset_weights()
        else:
            self.model.load_state_dict(model_state)

    def run_epoch(self, epoch_info: EpochInfo, loader: DatasetLoader):
        """ Run full epoch of learning """
        epoch_info.on_epoch_begin()

        lr = epoch_info.optimizer.param_groups[-1]['lr']
        print("|-------- Epoch {:06} Lr={:.6f} ----------|".format(epoch_info.global_epoch_idx, lr))

        self.train_epoch(epoch_info, loader)
        epoch_info.result_accumulator.freeze_results('train')

        self.validation_epoch(epoch_info, loader)
        epoch_info.result_accumulator.freeze_results('val')

        epoch_info.on_epoch_end()

    def train_epoch(self, epoch_info, loader: DatasetLoader, interactive=True):
        """ Run a single training epoch """
        self.train()

        if interactive:
            iterator = tqdm.tqdm(loader['train'], desc="Training", unit="iter", file=sys.stdout)
        else:
            iterator = loader['train']

        for batch_idx, datapoint in enumerate(iterator):
            batch_info = BatchInfo(epoch_info, batch_idx)
            batch_info['datapoint'] = datapoint

            batch_info.on_batch_begin('train')
            self.train_batch(batch_info, datapoint)
            batch_info.on_batch_end('train')

            iterator.set_postfix(loss=epoch_info.result_accumulator.intermediate_value('loss'))

    def validation_epoch(self, epoch_info, loader: DatasetLoader, interactive=True):
        """ Run a single evaluation epoch """
        self.eval()

        if interactive:
            iterator = tqdm.tqdm(loader['val'], desc="Training", unit="iter", file=sys.stdout)
        else:
            iterator = loader['val']

        with torch.no_grad():
            for batch_idx, datapoint in enumerate(iterator):
                batch_info = BatchInfo(epoch_info, batch_idx)
                batch_info['datapoint'] = datapoint

                batch_info.on_batch_begin('val')
                self.feed_batch(batch_info, datapoint)
                batch_info.on_batch_end('val')

    def feed_batch(self, batch_info, data):
        """ Run single batch of data """
        data = to_device(data, self.device)  # Move a data batch into the right device

        metrics = self.model.calculate_gradient(data)

        batch_info.update(metrics)

    def train_batch(self, batch_info, data):
        """ Train single batch of data """
        batch_info.optimizer.zero_grad()
        self.feed_batch(batch_info, data)

        if self.max_grad_norm is not None:
            batch_info['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=self.max_grad_norm
            )

        batch_info.optimizer.step()
