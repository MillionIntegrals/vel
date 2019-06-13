import vel.data as data


class Unsupervised(data.Augmentation):
    """ Simply transform supervised to an unsupervised dataset, cloning data to a target """
    def __init__(self):
        super().__init__('both', None)

    def __call__(self, x_data, y_data):
        return x_data, x_data


def create():
    return Unsupervised()
