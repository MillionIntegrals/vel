import torch
import torch.optim
import torch.nn.functional as F

from waterboy.internals.metrics.epoch_result import EpochResultAccumulator
from waterboy.internals.metrics.loss_metric import Loss


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs, optimizer, callbacks, log_frequency):
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.log_frequency = log_frequency

    def run(self, model, source, model_config, metrics):
        """ Run the command with supplied configuration """
        device = torch.device(model_config.device)
        model = model.to(device)
        metrics = [Loss()] + metrics

        for i in range(1, self.epochs+1):
            raise RuntimeError()
            print("|-------- Epoch {:06} ----------|".format(i))
            self.run_epoch(i, model, source, self.optimizer, metrics, device)

    def run_epoch(self, epoch_idx, model, source, optimizer, metrics, device):
        """ Run single epoch of training """
        result_accumulator = EpochResultAccumulator(epoch_idx, metrics)

        model.train()

        # TRAINING PART
        for batch_idx, (data, target) in enumerate(source.train_source):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            # No need for gradient calculations
            with torch.no_grad():
                result_accumulator.calculate(data, target, output, loss=loss)

            if batch_idx % self.log_frequency == 0:
                print('Train Epoch: {:04} [{:06}/{:06} ({:02.0f}%)]\t{}'.format(
                    epoch_idx,
                    batch_idx * len(data),
                    len(source.train_source.dataset), 100. * batch_idx / len(source.train_source),
                    result_accumulator.value_string())
                )

        print()
        print('Training:   {}'.format(result_accumulator.value_string()))
        result_accumulator.freeze_train_results()

        # EVALUATION PART
        model.eval()

        with torch.no_grad():
            for data, target in source.val_source:
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = F.nll_loss(output, target)

                result_accumulator.calculate(data, target, output, loss=loss)

        print('Validation: {}'.format(result_accumulator.value_string()))
        print()

        result_accumulator.freeze_validation_results()

        return result_accumulator.result()


def create(epochs, optimizer, callbacks=None, log_frequency=100):
    """ Simply train the model """
    import warnings
    warnings.filterwarnings("error")

    callbacks = callbacks or []

    return SimpleTrainCommand(epochs=epochs, optimizer=optimizer, callbacks=callbacks, log_frequency=log_frequency)
