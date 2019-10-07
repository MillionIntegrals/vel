class ModelSummary:
    """ Just print model summary """
    def __init__(self, model):
        self.model_factory = model

    def run(self, *args):
        """ Print model summary """
        model = self.model_factory.instantiate()
        model.summary()


def create(model):
    """ Vel factory function """
    return ModelSummary(model)
