import numpy as np

from .base_metric import BaseMetric


class ImageMetric(BaseMetric):
    """ Metric that logs an image """

    def metric_type(self) -> str:
        return 'image'


class RandomImageMetric(ImageMetric):
    """ Just pick a random image from the supplied list """

    def __init__(self, name, scope="general"):
        super().__init__(name, scope=scope)

        self.image = None

    def calculate(self, batch_info):
        batch = batch_info[self.name]

        if batch is not None:
            if len(batch.shape) > 3:
                image = batch[np.random.choice(batch.shape[0])]
            else:
                image = batch

            if image.shape[2] == 1:
                image = np.broadcast_to(image, shape=(image.shape[0], image.shape[1], 3))

            self.image = image

    def reset(self):
        self.image = None

    def value(self):
        return self.image

