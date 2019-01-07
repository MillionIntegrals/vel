class GenericFactory:
    """ Essentially a non-evaluated lambda function """
    def __init__(self, function, arguments=None):
        self.function = function
        self.arguments = arguments if arguments is not None else {}

    def instantiate(self, **kwargs):
        """ Create an instance """
        # Unpack both arguments
        return self.function(**{**kwargs, **self.arguments})
