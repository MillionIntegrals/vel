import attr
import typing

from vel.api import TrainingInfo


@attr.s(auto_attribs=True, frozen=True)
class MetricKey:
    """ Key for each metric """
    name: str
    scope: str
    dataset: typing.Optional[str] = None
    metric_type: str = 'scalar'

    def format(self):
        """ Format a metric key into a string """
        if self.dataset is None:
            return f"{self.scope}/{self.name}"
        else:
            return f"{self.dataset}:{self.scope}/{self.name}"


class BaseMetric:
    """ Base class for all the metrics """

    def __init__(self, name, scope="general"):
        self.name = name
        self.scope = scope

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        raise NotImplementedError

    def value(self):
        """ Return current value for the metric """
        raise NotImplementedError

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) -> None:
        """ Potentially store some metric state to the checkpoint """
        pass

    def prefix(self, prefix: str):
        """ Prepend a prefix to the name of the metric """
        self.name = f"{prefix}.{self.name}"
        return self

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) -> None:
        """ Potentially load some metric state from the checkpoint """
        pass

    def metric_type(self) -> str:
        """ Type of the metric """
        return 'scalar'
