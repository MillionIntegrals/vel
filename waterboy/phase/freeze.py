import waterboy.api.base as base


class FreezePhase(base.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, source):
        """ Freeze the model """
        model.freeze()


def create():
    """ Waterboy creation function """
    return FreezePhase()
