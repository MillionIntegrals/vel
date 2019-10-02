import torch

from vel.api import Network


def image_to_tensor(image: torch.Tensor) -> torch.Tensor:
    """ Convert pytorch image (b, w, h, c) into tensor (b, c, w, h) float32 """
    result = image.permute(0, 3, 1, 2).contiguous()

    if result.dtype == torch.uint8:
        result = result.type(torch.float) / 255.0
    else:
        result = result.type(torch.float)

    return result


class ImageToTensor(Network):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """

    def forward(self, image):
        return image_to_tensor(image)
