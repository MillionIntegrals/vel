class EnvRollerBase:
    """ Class generating environment rollouts """

    def rollout(self, batch_info, model) -> dict:
        """ Roll-out the environment and return it """
        raise NotImplementedError


# noinspection PyAbstractClass
class ReplayEnvRollerBase(EnvRollerBase):
    """ Class generating environment rollouts with experience replay """

    def sample(self, batch_info, batch_size) -> dict:
        """ Sample experience from replay buffer and return a batch """
        raise NotImplementedError


class EnvRollerFactory:
    """ Factory for env rollers """

    def instantiate(self, environment, device, settings) -> EnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError
