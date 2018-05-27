

class Augmentation:
    """ Base class for all data augmentations """
    def __init__(self, mode='x', tags=None):
        self.mode = mode
        self.tags = tags or ['train', 'val', 'test']

    def __call__(self, *args):
        """ Do the transformation """
        print(self)
        raise NotImplementedError

    def denormalize(self, *args):
        """ Operation reverse to normalization """
        if len(args) == 1:
            return args[0]
        else:
            return args
