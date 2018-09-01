class EnvRollerBase:
    """ Class generating environment rollouts """

    def rollout(self, model) -> dict:
        """ Calculate env rollout """
        raise NotImplementedError


# noinspection PyAbstractClass
class ReplayEnvRollerBase(EnvRollerBase):
    """ Class generating environment rollouts with experience replay """

    def rollout_replay(self, model) -> dict:
        raise NotImplementedError


class EnvRollerFactory:
    """ Factory for env rollers """

    def instantiate(self, environment, device, settings) -> EnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError
