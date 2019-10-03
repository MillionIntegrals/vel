from vel.api import OptimizedModel, VelOptimizer, OptimizerFactory, BatchInfo
from vel.rl.api import Rollout


class RlPolicy(OptimizedModel):
    """ Base class for reinforcement learning policies """

    def __init__(self, discount_factor: float):
        super().__init__()

        self.discount_factor = discount_factor

    def process_rollout(self, rollout: Rollout) -> Rollout:
        """ Process rollout for optimization before any chunking/shuffling  """
        return rollout

    def act(self, observation, state=None, deterministic=False) -> dict:
        """
        Make an action based on the observation from the environment.
        Returned dictionary must have 'actions' key that contains an action per
        each env in the observations
        """
        raise NotImplementedError

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return optimizer_factory.instantiate(parameters)

    def optimize(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """
        Perform one step of optimization of the policy based on provided rollout data
        :returns a dictionary of metrics
        """
        batch_info.optimizer.zero_grad()

        metrics = self.calculate_gradient(batch_info, rollout)

        opt_metrics = batch_info.optimizer.step()

        for key, value in opt_metrics.items():
            metrics[key] = value

        self.post_optimization_step(batch_info, rollout)

        return metrics

    def calculate_gradient(self, batch_info: BatchInfo, rollout: Rollout) -> dict:
        """
        Calculate gradient for given batch of training data.
        :returns a dictionary of metrics
        """
        raise NotImplementedError

    def post_optimization_step(self, batch_info: BatchInfo, rollout: Rollout):
        """ Optional operations to perform after optimization """
        pass

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    ####################################################################################################################
    # Utility Methods - that provide default implementations but may be short circuited by some implementations
    def action(self, observation, state=None, deterministic=False):
        """ Return policy action for given observation """
        return self.act(observation, state=state, deterministic=deterministic)['actions']
