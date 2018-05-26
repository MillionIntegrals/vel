import torch
import time

from waterboy.api.metrics.epoch_result import EpochResultAccumulator
from waterboy.api.progress_idx import ProgressIdx
from waterboy.callbacks.checkpoint import Checkpoint


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs, optimizer_fn, scheduler_fn, callbacks, log_frequency, checkpoint, model, source,
                 model_config):
        self.epochs = epochs
        self.callbacks = callbacks
        self.log_frequency = log_frequency
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.checkpoint = checkpoint
        self.model = model
        self.source = source
        self.model_config = model_config

    def run(self):
        """ Run the command with supplied configuration """
        model = self.model
        model_config = self.model_config
        source = self.source

        device = torch.device(model_config.device)
        model = model.to(device)

        optimizer_instance = self.optimizer_fn(model.parameters())

        callbacks = []

        checkpoint = Checkpoint(
            model_config,
            checkpoint_frequency=self.checkpoint.get('frequency', None),
            metric=self.checkpoint.get('metric', None),
            metric_mode=self.checkpoint.get('mode', 'min'),
        )

        last_epoch = checkpoint.last_epoch()

        if last_epoch > 0:
            checkpoint.load_model(last_epoch, model, optimizer_instance)
            print("Resuming training from epoch: {}".format(last_epoch+1))

            if checkpoint.metric is not None and model_config.provider.has_name('storage'):
                storage = model_config.provider.instantiate_by_name('storage')
                best_value, best_epoch = storage.best_metric(last_epoch, checkpoint.metric, checkpoint.metric_mode)
                checkpoint.best_value = best_value
                checkpoint.best_epoch = best_epoch

        callbacks.append(checkpoint)

        # Add storage if defined
        if model_config.provider.has_name('storage'):
            storage = model_config.provider.instantiate_by_name('storage')
            storage.clean(last_epoch)
            callbacks.append(storage)

        if self.scheduler_fn:
            callbacks.append(self.scheduler_fn(optimizer_instance))

        metrics = model.metrics()

        print("-" * 120)
        model.summary()
        print("-" * 120)
        print("Number of model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
        print("-" * 120)

        start_time = time.time()

        for callback in callbacks:
            callback.on_train_begin()

        for epoch_idx in range(1 + last_epoch, self.epochs+1):
            for callback in callbacks:
                callback.on_epoch_begin(epoch_idx)

            lr = optimizer_instance.param_groups[0]['lr']
            print("|-------- Epoch {:06} Lr={:.6f} ----------|".format(epoch_idx, lr))
            epoch_result = self.run_epoch(epoch_idx, model, source, optimizer_instance, metrics, device, callbacks)

            epoch_time = time.time() - start_time

            for callback in callbacks:
                callback.on_epoch_end(epoch_idx, epoch_time, epoch_result, model, optimizer_instance)

        for callback in callbacks:
            callback.on_train_end()

    def run_epoch(self, epoch_idx, model, source, optimizer, metrics, device, callbacks):
        """ Run single epoch of training """
        result_accumulator = EpochResultAccumulator(epoch_idx, metrics)

        model.train()

        # TRAINING PART
        for batch_idx, (data, target) in enumerate(source.train_source):
            progress_idx = ProgressIdx(epoch_idx, batch_idx, source.train_iterations_per_epoch())

            for callback in callbacks:
                callback.on_batch_begin(progress_idx)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output, loss = model.loss(data, target)

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

            for callback in callbacks:
                callback.on_batch_end(progress_idx, result_accumulator.value())

        print()
        print('Training:   {}'.format(result_accumulator.value_string()))
        result_accumulator.freeze_train_results()

        # EVALUATION PART
        model.eval()

        with torch.no_grad():
            for data, target in source.val_source:
                data, target = data.to(device), target.to(device)
                output, loss = model.loss(data, target)

                result_accumulator.calculate(data, target, output, loss=loss)

        print('Validation: {}'.format(result_accumulator.value_string()))
        print()

        result_accumulator.freeze_validation_results()

        return result_accumulator.result()


def create(epochs, optimizer, model, source, model_config, scheduler=None, callbacks=None, log_frequency=100, checkpoint=None):
    """ Simply train the model """
    callbacks = callbacks or []
    checkpoint = checkpoint or {}

    return SimpleTrainCommand(
        epochs=epochs, optimizer_fn=optimizer, scheduler_fn=scheduler,
        callbacks=callbacks, log_frequency=log_frequency, checkpoint=checkpoint,
        model=model, source=source, model_config=model_config
    )
