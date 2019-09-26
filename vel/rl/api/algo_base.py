
class AlgoBase:
    """ Base class for algo reinforcement calculations """

    def initialize(self, training_info, model, environment, device):
        """ Initialize algo from reinforcer settings """
        pass

    def process_rollout(self, batch_info, rollout):
        """ Process rollout for ALGO before any chunking/shuffling  """
        return rollout

    def optimize(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


class OptimizerAlgoBase(AlgoBase):
    """ RL algo that does a simple optimizer update """

    def calculate_gradient(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        raise NotImplementedError

    def post_optimization_step(self, batch_info, device, model, rollout):
        """ Steps to take after optimization has been done"""
        pass

    def optimize(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        batch_info.optimizer.zero_grad()

        batch_result = self.calculate_gradient(batch_info=batch_info, device=device, model=model, rollout=rollout)

        batch_info.optimizer.step(closure=None)

        self.post_optimization_step(batch_info, device, model, rollout)

        return batch_result

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []
