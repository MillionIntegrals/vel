from vel.api import TrainingInfo


class BaseMetric:
    """ Base class for all the metrics """

    def __init__(self, name):
        self.name = name

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

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) -> None:
        """ Potentially load some metric state from the checkpoint """
        pass
