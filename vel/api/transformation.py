class Transformation:
    """ Base class for all data augmentations """
    def __init__(self, tags=None):
        self.tags = ['train', 'val', 'test'] if tags is None else tags

        if isinstance(self.tags, str):
            self.tags = [self.tags]

    def initialize(self, source):
        """ Initialize transformation from source """
        pass

    def __call__(self, datapoint):
        """ Do the transformation """
        raise NotImplementedError

    def denormalize(self, datapoint):
        """ Operation reverse to normalization """
        return datapoint

    def denormalize_item(self, datapoint_item, coordinate):
        """ Denormalize only a single item of the datapoint """
        return datapoint_item


class ScopedTransformation(Transformation):
    """ Transformation applied only to certain keys of the datapoint """

    def __init__(self, scope=None, tags=None):
        super().__init__(tags)

        self.scope = ['x'] if scope is None else scope

        # If there is only one, we wrap it as a list
        if isinstance(self.scope, str):
            self.scope = [self.scope]

    def transform(self, value):
        """ Actual transformation code """
        raise NotImplementedError

    def denormalization_transform(self, value):
        """ Operation reverse to normalization """
        return value

    def __call__(self, datapoint):
        """ Do the transformation """
        for name in self.scope:
            datapoint[name] = self.transform(datapoint[name])

        return datapoint

    def denormalize(self, datapoint):
        """ Operation reverse to normalization """
        for name in self.scope:
            datapoint[name] = self.denormalization_transform(datapoint[name])

        return datapoint

    def denormalize_item(self, datapoint_item, coordinate):
        """ Denormalize only a single item of the datapoint """
        if coordinate in self.scope:
            return self.denormalization_transform(datapoint_item)
        else:
            return datapoint_item
