import importlib
import inspect

from vel.internals.parser import Variable
from vel.internals.generic_factory import GenericFactory


class Provider:
    """ Dependency injection resolver for the configuration file """
    def __init__(self, environment, instances=None, parameters=None):
        self.environment = environment

        self.parameters = parameters if parameters is not None else {}

        self.instances = {
            **(instances if instances is not None else {}),
            'vel_provider': self
        }

    def inject(self, name, value):
        """ Inject an object into the provider """
        self.instances[name] = value

    def resolve_parameters(self, func, extra_env=None):
        """ Resolve parameter dictionary for the supplied function """
        parameter_list = [
            (k, v.default == inspect.Parameter.empty) for k, v in inspect.signature(func).parameters.items()
        ]
        extra_env = extra_env if extra_env is not None else {}
        kwargs = {}

        for parameter_name, is_required in parameter_list:
            # extra_env is a 'local' object data defined in-place
            if parameter_name in extra_env:
                kwargs[parameter_name] = self.instantiate_from_data(extra_env[parameter_name])
                continue

            if parameter_name in self.instances:
                kwargs[parameter_name] = self.instances[parameter_name]
                continue

            if parameter_name in self.environment:
                kwargs[parameter_name] = self.instantiate_by_name(parameter_name)
                continue

            if is_required:
                funcname = f"{inspect.getmodule(func).__name__}.{func.__name__}"
                raise RuntimeError("Required argument '{}' cannot be resolved for function '{}'".format(
                    parameter_name, funcname
                ))

        return kwargs

    def resolve_and_call(self, func, extra_env=None):
        """ Resolve function arguments and call them, possibily filling from the environment """
        kwargs = self.resolve_parameters(func, extra_env=extra_env)
        return func(**kwargs)

    def instantiate_from_data(self, object_data):
        """ Instantiate object from the supplied data, additional args may come from the environment """
        if isinstance(object_data, dict) and 'name' in object_data:
            name = object_data['name']
            module = importlib.import_module(name)
            return self.resolve_and_call(module.create, extra_env=object_data)
        if isinstance(object_data, dict) and 'factory' in object_data:
            factory = object_data['factory']
            module = importlib.import_module(factory)
            params = self.resolve_parameters(module.create, extra_env=object_data)
            return GenericFactory(module.create, params)
        elif isinstance(object_data, dict):
            return {k: self.instantiate_from_data(v) for k, v in object_data.items()}
        elif isinstance(object_data, list):
            return [self.instantiate_from_data(x) for x in object_data]
        elif isinstance(object_data, Variable):
            return object_data.resolve(self.parameters)
        else:
            return object_data

    def render_configuration(self, configuration=None):
        """ Render variables in configuration object but don't instantiate anything """
        if configuration is None:
            configuration = self.environment

        if isinstance(configuration, dict):
            return {k: self.render_configuration(v) for k, v in configuration.items()}
        elif isinstance(configuration, list):
            return [self.render_configuration(x) for x in configuration]
        elif isinstance(configuration, Variable):
            return configuration.resolve(self.parameters)
        else:
            return configuration

    def has_name(self, object_name):
        """ Check if given name is available in the provider """
        return object_name in self.instances or object_name in self.environment

    def instantiate_by_name(self, object_name):
        """ Instantiate object from the environment, possibly giving some extra arguments """
        if object_name not in self.instances:
            instance = self.instantiate_from_data(self.environment[object_name])

            self.instances[object_name] = instance
            return instance
        else:
            return self.instances[object_name]

    def instantiate_by_name_with_default(self, object_name, default_value=None):
        """ Instantiate object from the environment, possibly giving some extra arguments """
        if object_name not in self.instances:
            if object_name not in self.environment:
                return default_value
            else:
                instance = self.instantiate_from_data(self.environment[object_name])

                self.instances[object_name] = instance
                return instance
        else:
            return self.instances[object_name]

    def get(self, name):
        """ Get object from given provider """
        return self.instantiate_by_name(name)
