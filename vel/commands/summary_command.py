from vel.api.base import Source


class ModelSummary:
    """ Just print model summary """
    def __init__(self, model, source: Source):
        self.model = model
        self.source = source

    def run(self, *args):
        """ Print model summary """
        if self.source is None:
            self.model.summary()
        else:
            x_data, y_data = next(iter(self.source.train_loader()))
            self.model.summary(input_size=x_data.shape[1:])


def create(model, source=None):
    """ Vel creation function """
    return ModelSummary(model, source)
