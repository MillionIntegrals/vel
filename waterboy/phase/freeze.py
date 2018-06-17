import waterboy.api.base as base


class FreezePhase(base.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Freeze the model """
        learner.model.freeze()


def create():
    """ Waterboy creation function """
    return FreezePhase()
