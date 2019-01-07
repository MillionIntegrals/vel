import vel.api as api


class FreezePhase(api.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, source):
        """ Freeze the model """
        model.freeze()


def create():
    """ Vel factory function """
    return FreezePhase()
