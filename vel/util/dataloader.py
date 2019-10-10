class IteratorDictWrapper:
    """ Transform dataset loader into a dictionary """

    def __init__(self, iterator, field_mapping):
        self.iterator = iterator
        self.field_mapping = field_mapping

    def __iter__(self):
        return map(self.map_values, iter(self.iterator))

    def __len__(self):
        return len(self.iterator)

    def map_values(self, item):
        """ Map iterator values into a dictionary """
        return {
            name: getattr(item, argument) for name, argument in self.field_mapping.items()
        }
