from .callback import Callback


class SchedulerFactory:
    """ Factory class for various schedulers """

    def instantiate(self, optimizer, last_epoch=-1) -> Callback:
        raise NotImplementedError
