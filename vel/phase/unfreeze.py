import vel.api.base as base


class UnfreezePhase(base.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, source):
        """ Freeze the model """
        model.unfreeze()


def create():
    """ Vel creation function """
    return UnfreezePhase()
