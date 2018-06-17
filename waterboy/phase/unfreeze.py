import waterboy.api.base as base


class UnfreezePhase(base.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Freeze the model """
        learner.model.unfreeze()


def create():
    """ Waterboy creation function """
    return UnfreezePhase()
