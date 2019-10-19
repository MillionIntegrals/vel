from vel.api import Transformation


class Unsupervised(Transformation):
    """ Simply transform supervised to an unsupervised dataset, cloning data to a target """
    def __call__(self, datapoint):
        datapoint['y'] = datapoint['x']
        return datapoint


def create():
    return Unsupervised()
