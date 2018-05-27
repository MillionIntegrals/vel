import matplotlib.pyplot as plt
import numpy as np


class AugmentationVisualizationCommand:
    """ Visualize augmentations """
    def __init__(self, source, samples, cases, seed=None):
        self.source = source
        self.samples = samples
        self.cases = cases
        self.seed = seed

    def run(self):
        """ Run the visualization """
        if self.seed is not None:
            np.random.seed(self.seed)

        dataset = self.source.train_dataset()
        num_samples = len(dataset)

        fig, ax = plt.subplots(self.cases, self.samples+1)

        selected_sample = np.sort(np.random.choice(num_samples, self.cases, replace=False))

        for i in range(self.cases):
            raw_image, _ = dataset.get_raw(selected_sample[i])

            ax[i, 0].imshow(raw_image)
            ax[i, 0].set_title("Original image")

            for j in range(self.samples):
                augmented_image, _ = dataset[selected_sample[i]]
                augmented_image = dataset.denormalize(augmented_image)
                ax[i, j+1].imshow(augmented_image)

        plt.show()


def create(source, samples, cases, seed=None):
    """ Visualize augmentations in the source """
    return AugmentationVisualizationCommand(source, samples, cases, seed)
