"""
Slightly modified version of:
https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
"""

import numbers
import random

import vel.api.data as data


class RandomCrop(data.Augmentation):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, padding_mode='constant', pad_if_needed=False, mode='x', tags=None):
        super().__init__(mode, tags)

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding
        self.padding_mode = padding_mode
        self.padding_mode_cv = data.mode_to_cv2(self.padding_mode)
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h, *_ = img.shape

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = data.pad(img, self.padding, mode=self.padding_mode_cv)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = data.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0), mode=self.padding_mode_cv)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = data.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)), mode=self.padding_mode_cv)

        i, j, h, w = self.get_params(img, self.size)

        return data.crop(img, j, i, w, h)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def create(width, height, padding=0, padding_mode='constant', mode='x', tags=None):
    """ Vel factory function """
    return RandomCrop(size=(width, height), padding=padding, padding_mode=padding_mode, mode=mode, tags=tags)
