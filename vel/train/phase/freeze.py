import vel.train as train


class FreezePhase(train.EmptyTrainPhase):
    """ Freeze the model """
    def __init__(self, groups=None):
        self.groups = groups

    def set_up_phase(self, training_info, model, loader):
        """ Freeze the model """
        model.freeze(groups=self.groups)


def create(groups=None):
    """ Vel factory function """
    return FreezePhase(groups)
