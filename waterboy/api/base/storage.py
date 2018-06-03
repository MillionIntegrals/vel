import typing


class Storage:
    """ Base class for waterboy storage implementations """

    def set_checkpoint_strategy(self, new_checkpoint_strategy):
        raise NotImplementedError

    def resume_learning(self, model) -> (int, typing.Union[dict, None]):
        raise NotImplementedError

    def streaming_callbacks(self) -> list:
        """ Lift of callbacks for live streaming results """
        return []

    def restore(self, hidden_state):
        """ Restore optimizer and callbacks from hidden state """
        pass

    def get_frame(self):
        """ Get a frame of metrics from backend """
        raise NotImplementedError
