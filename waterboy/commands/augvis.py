import matplotlib.pyplot as plt
import numpy as np


class AugmentationVisualizationCommand:
    """ Visualize augmentations """
    def __init__(self, source, samples):
        self.source = source
        self.samples = samples

    def run(self):
        """ Run the visualization """
        dataset = self.source.train_source.dataset
        num_samples = len(self.source.train_source.dataset)

        selected_sample = np.sort(np.random.choice(num_samples, self.samples, replace=False))

        fig, ax = plt.subplots(self.samples, 2)
        transform = dataset.transform

        for i in range(self.samples):
            # A hack for now, later I'll fork my own dataset API
            dataset.transform = None

            raw_data = np.array(dataset[selected_sample[i]][0])

            dataset.transform = transform

            transformed_data = np.transpose(dataset[selected_sample[i]][0].numpy(), (1, 2, 0))

            ax[i, 0].imshow(raw_data)
            ax[i, 1].imshow(transformed_data)

        plt.show()


def create(source, samples):
    """ Visualize augmentations in the source """
    return AugmentationVisualizationCommand(source, samples)
