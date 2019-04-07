from vel.api import EpochInfo, TrainingInfo, Model


class Storage:
    """ Base class for Vel storage implementations """

    def last_epoch_idx(self) -> int:
        """ Return last checkpointed epoch idx for given configuration. Returns 0 if no results have been stored """
        raise NotImplementedError

    def load(self, train_info: TrainingInfo) -> (dict, dict):
        """
        Resume learning process and return loaded hidden state dictionary
        """
        raise NotImplementedError

    def reset(self, configuration: dict) -> None:
        """
        Whenever there was anything stored in the database or not, purge previous state and start
        new training process from scratch.
        """
        raise NotImplementedError

    def streaming_callbacks(self) -> list:
        """ Lift of callbacks for live streaming results """
        return []

    def get_metrics_frame(self):
        """ Get a frame of metrics from backend """
        raise NotImplementedError

    def checkpoint(self, epoch_info: EpochInfo, model: Model):
        """ When epoch is done, we persist the training state """
        raise NotImplementedError
