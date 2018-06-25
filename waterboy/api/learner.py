import torch
import tqdm
import sys

from .progress_idx import BatchIdx
from .metrics import EpochResultAccumulator


class Learner:
    """ Manages training process of a single model """

    def __init__(self, device, model):
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

    def train_batch(self, data, target, optimizer, result_accumulator=None):
        """ Run single batch of data """
        data, target = data.to(self.device), target.to(self.device)

        optimizer.zero_grad()

        output, loss = self.model.loss(data, target)

        loss.backward()
        optimizer.step()

        # No need for gradient calculations
        if result_accumulator is not None:
            with torch.no_grad():
                result_accumulator.calculate(data, target, output, context={
                    'loss': loss, 'model': self.model, 'optimizer': optimizer
                })

    def train_epoch(self, epoch_idx, source, optimizer, callbacks=None, result_accumulator=None):
        """ Run a single training epoch """
        callbacks = callbacks or []
        self.train()

        iterator = tqdm.tqdm(source.train_loader, desc="Training", unit="iter", file=sys.stdout)

        for batch_idx, (data, target) in enumerate(iterator):
            progress_idx = BatchIdx(epoch_idx, batch_idx, source.train_iterations_per_epoch())

            for callback in callbacks:
                callback.on_batch_begin(progress_idx)

            self.train_batch(data, target, optimizer, result_accumulator)

            iterator.set_postfix(loss=result_accumulator.intermediate_value('loss'))

            for callback in callbacks:
                callback.on_batch_end(progress_idx, result_accumulator.value(), optimizer)

    def eval_epoch(self, epoch_idx, source, callbacks=None, result_accumulator=None):
        """ Run a single evaluation epoch """
        self.eval()

        for callback in callbacks:
            callback.on_validation_begin(epoch_idx)

        with torch.no_grad():
            if source.is_tta_enabled():
                tta_accumulator = source.tta_accumulator(result_accumulator)

                for data, target in tqdm.tqdm(source.val_tta_loader, desc="Validation", unit="iter", file=sys.stdout):
                    data, target = data.to(self.device), target.to(self.device)
                    output, loss = self.model.loss(data, target)

                    tta_accumulator.calculate(data, target, output, context={'loss': loss})
            else:
                for data, target in tqdm.tqdm(source.val_loader, desc="Validation", unit="iter", file=sys.stdout):
                    data, target = data.to(self.device), target.to(self.device)
                    output, loss = self.model.loss(data, target)

                    result_accumulator.calculate(data, target, output, context={'loss': loss})

        for callback in callbacks:
            callback.on_validation_end(epoch_idx, result_accumulator.value())

    def run_epoch(self, epoch_idx, metrics, source, optimizer, callbacks):
        """ Run full epoch of learning """
        result_accumulator = EpochResultAccumulator(epoch_idx, metrics)

        for callback in callbacks:
            callback.on_epoch_begin(epoch_idx)

        lr = optimizer.param_groups[-1]['lr']
        print("|-------- Epoch {:06} Lr={:.6f} ----------|".format(epoch_idx.global_epoch_idx, lr))

        self.train_epoch(epoch_idx, source, optimizer, callbacks, result_accumulator=result_accumulator)
        result_accumulator.freeze_train_results()

        self.eval_epoch(epoch_idx, source, callbacks, result_accumulator=result_accumulator)
        result_accumulator.freeze_validation_results()

        epoch_result = result_accumulator.result()

        for callback in callbacks:
            callback.on_epoch_end(epoch_idx, epoch_result)

        return epoch_result
