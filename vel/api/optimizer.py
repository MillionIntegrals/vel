import collections
import itertools as it
import typing

from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from vel.api.callback import Callback
from vel.api.scheduler import SchedulerFactory
from vel.exception import VelException
from vel.metric import DefaultAveragingNamedMetric
from vel.util.datastructure import flatten_dict


class VelOptimizer:
    """ Vel optimizer interface """

    def get_lr(self) -> float:
        """ Return current learning rate of the optimizer """
        raise NotImplementedError

    def set_lr(self, lr: float):
        """ Set current learning rate of the optimizer """
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    def step(self, closure=None) -> dict:
        raise NotImplementedError

    def add_param_group(self, param_group: dict) -> None:
        raise NotImplementedError

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return []

    def create_scheduler(self, scheduler_factory: SchedulerFactory, last_epoch: int = -1) -> [Callback]:
        """ Create a scheduler instance for this optimizer """
        raise NotImplementedError


class VelOptimizerProxy(VelOptimizer):
    """ Proxy PyTorch optimizer into a Vel optimizer """
    def __init__(self, optimizer: Optimizer, group_names: [str], max_grad_norm: typing.Optional[float] = None):
        self.optimizer = optimizer
        self.group_names = group_names
        self.max_grad_norm = max_grad_norm

        if 'default' in self.group_names:
            self.main_idx = self.group_names.index('default')
        else:
            self.main_idx = len(self.group_names) - 1

        assert len(self.optimizer.param_groups) == len(self.group_names), \
            "There must be equal number of parameter groups and group names"

        self.initial_lrs = [x['lr'] for x in self.optimizer.param_groups]

    def get_lr(self) -> float:
        """ Return current learning rate of the optimizer """
        return self.optimizer.param_groups[self.main_idx]['lr']

    def set_lr(self, lr: float):
        """ Set current learning rate of the optimizer """
        if isinstance(lr, list):
            for group_lr, param_group in zip(lr, self.optimizer.param_groups):
                param_group['lr'] = group_lr
        elif isinstance(lr, dict):
            for idx, name in enumerate(self.group_names):
                self.optimizer.param_groups[idx]['lr'] = lr[name]
        else:
            canonical_lr = self.initial_lrs[0]

            for idx, param_group in enumerate(self.optimizer.param_groups):
                opt_lr = self.initial_lrs[idx] / canonical_lr * lr
                param_group['lr'] = opt_lr

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, closure=None) -> dict:
        # TODO(jerry): potentially allow max_grad_norm being a list?
        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(
                parameters=it.chain.from_iterable(pg['params'] for pg in self.optimizer.param_groups),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step(closure)
            return {'grad_norm': grad_norm}
        else:
            self.optimizer.step(closure)
            return {}

    def add_param_group(self, param_group: dict) -> None:
        self.optimizer.add_param_group(param_group)

    def metrics(self) -> list:
        """ Set of metrics for this model """
        if self.max_grad_norm is not None:
            return [
                DefaultAveragingNamedMetric('grad_norm', scope="opt", defaut_value=0.0)
            ]
        else:
            return []

    def create_scheduler(self, scheduler_factory: SchedulerFactory, last_epoch: int = -1) -> [Callback]:
        """ Create a scheduler instance for this optimizer """
        return [scheduler_factory.instantiate(self.optimizer, last_epoch=last_epoch)]


class VelMultiOptimizer(VelOptimizer):
    """ Optimizer that wraps several individual optimizers """

    def __init__(self, optimizers: typing.Dict[str, VelOptimizer], canonical_name: typing.Optional[str] = None):
        self.optimizers = optimizers

        # Canonical, chosen optimizer
        if canonical_name is None:
            self.canonical_name = list(optimizers.keys())[0]
        else:
            self.canonical_name = canonical_name

        self.initial_lrs = {
            name: optimizer.get_lr()
            for name, optimizer in self.optimizers.items()
        }

    def __getitem__(self, item):
        return self.optimizers[item]

    def get_lr(self) -> float:
        return self.optimizers[self.canonical_name].get_lr()

    def set_lr(self, lr: float):
        if isinstance(lr, list):
            # TODO: implement
            raise NotImplementedError
        elif isinstance(lr, dict):
            # TODO: implement
            raise NotImplementedError
        else:
            canonical_lr = self.initial_lrs[self.canonical_name]

            for name, optimizer in self.optimizers.items():
                opt_lr = self.initial_lrs[name] / canonical_lr * lr
                optimizer.set_lr(opt_lr)

    def state_dict(self) -> dict:
        output = {}

        for name, optimizer in self.optimizers.items():
            output[name] = optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[name])

    def zero_grad(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def step(self, closure=None) -> dict:
        output = {}

        for name, optimizer in self.optimizers.items():
            metrics = optimizer.step()
            flatten_dict(metrics, output, name)

        return output

    def create_scheduler(self, scheduler_factory: SchedulerFactory, last_epoch: int = -1) -> [Callback]:
        """ Create a scheduler instance for this optimizer """
        return [
            scheduler_factory.instantiate(optimizer, last_epoch=last_epoch)
            for optimizer in self.optimizers.values()
        ]

    def add_param_group(self, param_group: dict) -> None:
        raise VelException("Unsupported operation")

    def metrics(self) -> list:
        """ Set of metrics for this model """
        # TODO(jerry): aggregate metrics
        return []


class OptimizerFactory:
    """ Base class for optimizer factories """
    def __init__(self):
        self.parameter_groups = None

    def with_parameter_groups(self, parameter_groups=None):
        """ Set `parameter_groups` for this factory """
        self.parameter_groups = parameter_groups
        return self

    def preprocess(self, parameters):
        """ Preprocess given parameters input into proper optimizer parameter groups, with their names """
        parameters = list(parameters)

        # Make sure parameters have right format
        if parameters:
            if not isinstance(parameters[0], collections.Sequence) or not isinstance(parameters[0][0], str):
                parameters = [("default", parameters)]

        groups = collections.defaultdict(list)

        for name, group in parameters:
            group = [x for x in group if x.requires_grad]
            if group:  # Must have at least 1 element
                groups[name].extend(group)

        group_names = []
        sorted_groups = []

        for name in sorted(groups.keys()):
            parameter_group = {
                'params': groups[name]
            }

            if self.parameter_groups and name in self.parameter_groups:
                parameter_group.update(self.parameter_groups[name])

            sorted_groups.append(parameter_group)
            group_names.append(name)

        return sorted_groups, group_names

    def instantiate(self, parameters) -> VelOptimizer:
        """ Instantiate VelOptimizer for iterable of parameters or iterable of (parameter, group) """
        raise NotImplementedError

    def instantiate_multi(self, parameter_dict: dict) -> VelMultiOptimizer:
        od = collections.OrderedDict()

        for name, value in parameter_dict.items():
            od[name] = self.instantiate(value)

        return VelMultiOptimizer(od)
