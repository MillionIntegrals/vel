import vel.train as train


class UnfreezePhase(train.EmptyTrainPhase):
    """ Freeze the model """

    def set_up_phase(self, training_info, model, loader):
        """ Freeze the model """
        model.unfreeze()


def create():
    """ Vel factory function """
    return UnfreezePhase()
