import torch

from waterboy.internals.metrics.epoch_result import EpochResultAccumulator


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs, optimizer_fn, scheduler_fn, callbacks, log_frequency):
        self.epochs = epochs
        self.callbacks = callbacks
        self.log_frequency = log_frequency
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

    def run(self, model, source, model_config):
        """ Run the command with supplied configuration """
        device = torch.device(model_config.device)
        model = model.to(device)

        optimizer_instance = self.optimizer_fn(model.parameters())

        if self.scheduler_fn:
            scheduler_instance = self.scheduler_fn(optimizer_instance)
        else:
            scheduler_instance = None

        metrics = model.metrics()

        print("-" * 120)
        model.summary()
        print("-" * 120)
        print("Number of model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
        print("-" * 120)

        for i in range(1, self.epochs+1):
            if scheduler_instance is not None:
                scheduler_instance.pre_epoch_step(i)
                lr = scheduler_instance.get_lr()[0]
            else:
                raise NotImplementedError

            print("|-------- Epoch {:06} Lr={:.6f} ----------|".format(i, lr))
            epoch_result = self.run_epoch(i, model, source, optimizer_instance, metrics, device)

            if scheduler_instance is not None:
                scheduler_instance.post_epoch_step(epoch_result)

    def run_epoch(self, epoch_idx, model, source, optimizer, metrics, device):
        """ Run single epoch of training """
        result_accumulator = EpochResultAccumulator(epoch_idx, metrics)

        model.train()

        # TRAINING PART
        for batch_idx, (data, target) in enumerate(source.train_source):
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


def create(epochs, optimizer, scheduler=None, callbacks=None, log_frequency=100):
    """ Simply train the model """
    import warnings
    warnings.filterwarnings("error")

    callbacks = callbacks or []

    return SimpleTrainCommand(
        epochs=epochs, optimizer_fn=optimizer, scheduler_fn=scheduler,
        callbacks=callbacks, log_frequency=log_frequency
    )
