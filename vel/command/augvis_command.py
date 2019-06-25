import matplotlib.pyplot as plt
import numpy as np

from vel.data import Loader


class AugmentationVisualizationCommand:
    """ Visualize augmentations """
    def __init__(self, loader: Loader, samples, cases):
        self.loader = loader
        self.samples = samples
        self.cases = cases

    def run(self):
        """ Run the visualization """
        dataset = self.loader.transformed_source.train
        num_samples = len(dataset)

        fig, ax = plt.subplots(self.cases, self.samples+1)

        selected_sample = np.sort(np.random.choice(num_samples, self.cases, replace=False))

        for i in range(self.cases):
            raw_image = dataset.get_raw(selected_sample[i])['x']

            ax[i, 0].imshow(raw_image)
            ax[i, 0].set_title("Original image")

            for j in range(self.samples):
                augmented_datapoint = dataset[selected_sample[i]]
                denormalized_datapoint = dataset.denormalize(augmented_datapoint)
                ax[i, j+1].imshow(np.clip(denormalized_datapoint['x'], 0.0, 1.0))

        plt.show()


def create(loader: Loader, samples: int, cases: int):
    """ Vel factory function """
    return AugmentationVisualizationCommand(loader, samples, cases)
