import torch.nn.utils


class AlgoBase:
    """ Base class for policy gradient calculations """
    def initialize(self, settings, model, environment, device):
        """ Initialize policy gradient from reinforcer settings """
        pass

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []

    def _clip_gradients(self, batch_info, model, max_grad_norm):
        """ Clip gradients to a given maximum length """
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=max_grad_norm
            )
        else:
            grad_norm = 0.0

        if 'sub_batch_data' in batch_info:
            batch_info['sub_batch_data'][-1]['grad_norm'] = grad_norm
        else:
            batch_info['grad_norm'] = grad_norm


class OptimizerAlgoBase(AlgoBase):
    """ Policy gradient that does a simple optimizer update """
    def __init__(self, max_grad_norm):
        self.max_grad_norm = max_grad_norm

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        raise NotImplementedError

    def post_optimization_step(self, batch_info, device, model, rollout):
        """ Steps to take after optimization has been done"""
        pass

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        batch_info.optimizer.zero_grad()

        loss = self.calculate_loss(batch_info=batch_info, device=device, model=model, rollout=rollout)

        loss.backward()

        self._clip_gradients(batch_info, model, self.max_grad_norm)

        batch_info.optimizer.step(closure=None)

        self.post_optimization_step(batch_info, device, model, rollout)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []
