import os
import yaml

from vel.exceptions import VelException


class Dummy:
    """ Dummy instance placeholder """
    pass


DUMMY_VALUE = Dummy()


class Variable:
    """ Project configuration variable to be popupated from command line """
    @classmethod
    def parameter_constructor(cls, loader, node):
        """ Construct variable instance from yaml node """
        value = loader.construct_scalar(node)

        if isinstance(value, str):
            if '=' in value:
                (varname, varvalue) = Parser.parse_equality(value)
                return cls(varname, varvalue)
            else:
                return cls(value)
        else:
            return cls(value)

    def __init__(self, name, default_value=DUMMY_VALUE):
        self.name = name
        self.default_value = default_value

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def resolve(self, parameters):
        """ Resolve given variable """
        raise NotImplementedError


class Parameter(Variable):
    """ Parameter that gets propagated from command line options """
    def resolve(self, parameters):
        """ Resolve given variable """
        if self.default_value == DUMMY_VALUE:
            if self.name in parameters:
                return parameters[self.name]
            else:
                raise VelException(f"Undefined parameter: {self.name}")
        else:
            return parameters.get(self.name, self.default_value)


class EnvironmentVariable(Variable):
    """ Parameter that gets propagated from environment """
    def resolve(self, _):
        """ Resolve given variable """
        if self.default_value == DUMMY_VALUE:
            if self.name in os.environ:
                return os.environ[self.name]
            else:
                raise VelException(f"Undefined environment variable: {self.name}")
        else:
            return os.environ.get(self.name, self.default_value)


class Parser:
    """ Parse configuration values """
    IS_LOADED = False

    @classmethod
    def register(cls):
        """ Register variable handling in YAML """
        if not cls.IS_LOADED:
            cls.IS_LOADED = True

            yaml.add_constructor('!param', Parameter.parameter_constructor, Loader=yaml.SafeLoader)
            yaml.add_constructor('!env', EnvironmentVariable.parameter_constructor, Loader=yaml.SafeLoader)

    @classmethod
    def parse(cls, stream):
        """ Parse the stream into a Python object """
        cls.register()
        return yaml.safe_load(stream)

    @classmethod
    def parse_equality(cls, equality_string):
        """ Parse some simple equality statements """
        cls.register()
        assert '=' in equality_string, "There must be an '=' sign in the equality"
        [left_side, right_side] = equality_string.split('=', 1)

        left_side_value = yaml.safe_load(left_side.strip())
        right_side_value = yaml.safe_load(right_side.strip())

        assert isinstance(left_side_value, str), "Left side of equality must be a string"

        return left_side_value, right_side_value
