import torch

from vel.api import BackboneModel, ModelFactory


class ImageToTensor(BackboneModel):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """

    def __init__(self):
        super().__init__()

    def reset_weights(self):
        pass

    def forward(self, image):
        result = image.permute(0, 3, 1, 2).contiguous()

        if result.dtype == torch.uint8:
            result = result.type(torch.float) / 255.0
        else:
            result = result.type(torch.float)

        return result


def create():
    """ Vel factory function """
    return ModelFactory.generic(ImageToTensor)


# Scripting interface
ImageToTensorFactory = create
