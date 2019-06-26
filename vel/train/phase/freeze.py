import vel.train as train


class FreezePhase(train.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, loader):
        """ Freeze the model """
        model.freeze()


def create():
    """ Vel factory function """
    return FreezePhase()
