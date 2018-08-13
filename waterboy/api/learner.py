import torch
import tqdm
import sys

from .info import BatchInfo, EpochInfo


class Learner:
    """ Manages training process of a single model """
    def __init__(self, device: torch.device, model):
        self.device = device
        self.model = model.to(device)

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

    def initialize_training(self):
        """ Prepare for training """
        self.model.reset_weights()

    def run_epoch(self, epoch_info: EpochInfo, source):
        """ Run full epoch of learning """
        for callback in epoch_info.training_info.callbacks:
            callback.on_epoch_begin(epoch_info)

        lr = epoch_info.optimizer.param_groups[-1]['lr']
        print("|-------- Epoch {:06} Lr={:.6f} ----------|".format(epoch_info.global_epoch_idx, lr))

        self.train_epoch(epoch_info, source)
        epoch_info.result_accumulator.freeze_results('train')

        self.validation_epoch(epoch_info, source)
        epoch_info.result_accumulator.freeze_results('val')

        epoch_info.freeze_epoch_result()

        for callback in epoch_info.callbacks:
            callback.on_epoch_end(epoch_info)

    def train_epoch(self, epoch_info, source):
        """ Run a single training epoch """
        self.train()

        iterator = tqdm.tqdm(source.train_loader, desc="Training", unit="iter", file=sys.stdout)

        for batch_idx, (data, target) in enumerate(iterator):
            batch_info = BatchInfo(epoch_info, batch_idx)

            for callback in epoch_info.callbacks:
                callback.on_batch_begin(batch_info)

            self.train_batch(batch_info, data, target)

            for callback in epoch_info.callbacks:
                callback.on_batch_end(batch_info)

            epoch_info.result_accumulator.calculate(batch_info)

            iterator.set_postfix(loss=epoch_info.result_accumulator.intermediate_value('loss'))

    def validation_epoch(self, epoch_info, source):
        """ Run a single evaluation epoch """
        self.eval()

        iterator = tqdm.tqdm(source.val_loader, desc="Validation", unit="iter", file=sys.stdout)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(iterator):
                batch_info = BatchInfo(epoch_info, batch_idx)

                for callback in epoch_info.callbacks:
                    callback.on_validation_batch_begin(batch_info)

                self.feed_batch(batch_info, data, target)

                for callback in epoch_info.callbacks:
                    callback.on_validation_batch_end(batch_info)

                epoch_info.result_accumulator.calculate(batch_info)

    def feed_batch(self, batch_info, data, target):
        """ Run single batch of data """
        data, target = data.to(self.device), target.to(self.device)
        output, loss = self.model.loss(data, target)

        # Store extra batch information for calculation of the statistics
        batch_info['data'] = data
        batch_info['target'] = target
        batch_info['output'] = output
        batch_info['loss'] = loss

        return loss

    def train_batch(self, batch_info, data, target):
        """ Train single batch of data """
        batch_info.optimizer.zero_grad()
        loss = self.feed_batch(batch_info, data, target)
        loss.backward()
        batch_info.optimizer.step()

