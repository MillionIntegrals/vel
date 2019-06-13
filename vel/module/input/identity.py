from vel.api import BackboneModel, ModelFactory


class Identity(BackboneModel):
    """ Identity transformation that doesn't do anything """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def reset_weights(self):
        pass


def create():
    """ Vel factory function """
    def instantiate(**_):
        return Identity()

    return ModelFactory.generic(instantiate)


# Scripting interface
IdentityFactory = create
