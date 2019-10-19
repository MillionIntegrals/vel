import numpy as np
import torch
import tqdm

from vel.api import TrainingInfo


class VaeNllCommand:
    """ Calculate NLL for the VAE using importance sampling """
    def __init__(self, model_config, model_factory, loader, storage, max_batch: int, samples: int):
        self.model_config = model_config
        self.model_factory = model_factory
        self.loader = loader
        self.storage = storage

        self.max_batch = max_batch
        self.samples = samples

    @torch.no_grad()
    def run(self):
        device = self.model_config.torch_device()
        model = self.model_factory.instantiate().to(device)

        start_epoch = self.storage.last_epoch_idx()

        training_info = TrainingInfo(start_epoch_idx=start_epoch)

        model_state, hidden_state = self.storage.load(training_info)
        model.load_state_dict(model_state)

        model.eval()

        validation_dataset = self.loader.source.validation

        results = []

        # Always take at least one
        batch_size = max(self.max_batch // self.samples, 1)

        for i in tqdm.trange(validation_dataset.num_batches(batch_size)):
            batch = validation_dataset.get_batch(i, batch_size)['x'].to(self.model_config.device)
            nll = model.nll(batch, num_posterior_samples=self.samples)

            results.append(nll.cpu().numpy())

        full_results = np.concatenate(results)

        print("NLL: {:.2f}".format(np.mean(full_results)))


def create(model_config, model, loader, storage, max_batch: int = 1024, samples: int = 100):
    """ Vel factory function """
    return VaeNllCommand(model_config, model, loader, storage, max_batch=max_batch, samples=samples)
