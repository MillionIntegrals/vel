class ModelSummary:
    """ Just print model summary """
    def __init__(self, model, vec_env):
        self.model_factory = model
        self.vec_env = vec_env

    def run(self, *args):
        """ Print model summary """
        env = self.vec_env.instantiate(parallel_envs=1, seed=1)
        model = self.model_factory.instantiate(action_space=env.action_space, observation_space=env.observation_space)
        model.summary()


def create(model, vec_env):
    """ Vel factory function """
    return ModelSummary(model, vec_env)
