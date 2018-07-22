import importlib
import inspect


class Provider:
    """ Dependency injection resolver for the configuration file """
    def __init__(self, environment, instances):
        self.environment = environment

        self.instances = {
            **(instances or {}),
            'pp_provider': self
        }

    def inject(self, name, value):
        """ Inject an object into the provider """
        self.instances[name] = value

    def resolve_and_call(self, func, extra_env=None):
        """ Resolve function arguments and call them, possibily filling from the environment """
        parameter_list = [(k, v.default == inspect.Parameter.empty) for k, v in inspect.signature(func).parameters.items()]
        extra_env = extra_env or {}
        kwargs = {}

        for parameter_name, is_required in parameter_list:
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

        return func(**kwargs)

    def instantiate_from_data(self, object_data):
        """ Instantiate object from the supplied data, additional args may come from the environment """
        if isinstance(object_data, dict) and 'name' in object_data:
            name = object_data['name']
            module = importlib.import_module(name)
            return self.resolve_and_call(module.create, extra_env=object_data)
        elif isinstance(object_data, list):
            return [self.instantiate_from_data(x) for x in object_data]
        else:
            return object_data

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
