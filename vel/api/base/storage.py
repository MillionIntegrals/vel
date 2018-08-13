import typing

from vel.api import EpochInfo
from vel.api.base import Model


class Storage:
    """ Base class for Vel storage implementations """

    def resume_learning(self, model) -> (int, typing.Union[dict, None]):
        raise NotImplementedError

    def streaming_callbacks(self) -> list:
        """ Lift of callbacks for live streaming results """
        return []

    def restore(self, hidden_state: dict):
        """ Restore optimizer and callbacks from hidden state """
        pass

    def get_frame(self):
        """ Get a frame of metrics from backend """
        raise NotImplementedError

    def checkpoint(self, epoch_info: EpochInfo, model: Model, state_dict: dict=None):
        """ When epoch is done, we persist the training state """
        raise NotImplementedError
