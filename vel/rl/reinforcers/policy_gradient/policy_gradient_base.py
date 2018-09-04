import torch.nn.utils


class PolicyGradientBase:
    """ Base class for policy gradient calculations """
    def initialize(self, settings):
        """ Initialize policy gradient from reinforcer settings """
        pass

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


class OptimizerPolicyGradientBase(PolicyGradientBase):
    """ Policy gradient that does a simple optimizer update """
    def __init__(self, max_grad_norm):
        self.max_grad_norm = max_grad_norm

    def calculate_loss(self, batch_info, device, model, rollout):
        """ Calculate loss of the supplied rollout """
        raise NotImplementedError

    def optimizer_step(self, batch_info, device, model, rollout):
        """ Single optimization step for a model """
        batch_info.optimizer.zero_grad()

        loss = self.calculate_loss(batch_info=batch_info, device=device, model=model, rollout=rollout)

        loss.backward()

        # Gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=self.max_grad_norm
            )

            batch_info['gradient_norms'].append(grad_norm)

        batch_info.optimizer.step(closure=None)

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []
