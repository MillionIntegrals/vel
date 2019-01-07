import vel.api as api


class UnfreezePhase(api.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, source):
        """ Freeze the model """
        model.unfreeze()


def create():
    """ Vel factory function """
    return UnfreezePhase()
