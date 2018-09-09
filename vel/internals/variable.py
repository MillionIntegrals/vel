import yaml


class Variable:
    """ Project configuration variable to be popupated from command line """
    @staticmethod
    def variable_constructor(loader, node):
        """ Construct variable instance from yaml node """
        value = loader.construct_scalar(node)
        return Variable(value)

    @staticmethod
    def register():
        """ Register variable handling in YAML """
        yaml.add_constructor('!var', Variable.variable_constructor, Loader=yaml.SafeLoader)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Variable(name={self.name})"
