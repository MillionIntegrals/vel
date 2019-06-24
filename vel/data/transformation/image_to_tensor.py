import numpy as np

import torchvision.transforms.functional as F

from vel.api.transformation import ScopedTransformation


class ImageToTensor(ScopedTransformation):
    """ Convert image array to a tensor """
    def transform(self, value):
        # First let's make sure it's actually a numpy array
        value = np.asarray(value)

        if len(value.shape) == 2:
            # If the image has only one channel, it still needs to be specified
            value = value.reshape(value.shape[0], value.shape[1], 1)

        return F.to_tensor(value)

    def denormalization_transform(self, value):
        image_array = np.transpose(value.numpy(), (1, 2, 0))

        if len(image_array.shape) == 3 and image_array.shape[-1] == 1:
            return image_array[:, :, 0]

        return image_array


def create(mode='x', tags=None):
    """ Vel factory function """
    return ImageToTensor(mode, tags)
