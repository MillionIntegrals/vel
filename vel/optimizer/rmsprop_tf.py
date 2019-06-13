import torch.optim

from vel.api import OptimizerFactory, Model


class RMSpropTF(torch.optim.Optimizer):
    """Implements RMSprop algorithm. A TensorFlow version with epsilon under the square root

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # ANOTHER LINE I'VE CHANGED
                    state['square_avg'] = torch.ones_like(p.data)

                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    # THIS LINE IS EVERYTHING THAT I CHANGED IN THIS OPTIMIZER
                    # avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt()
                else:
                    # THIS LINE IS EVERYTHING THAT I CHANGED IN THIS OPTIMIZER
                    # avg = square_avg.sqrt().add_(group['eps'])
                    avg = square_avg.add(group['eps']).sqrt()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss


class RMSpropTFFactory(OptimizerFactory):
    """ RMSprop optimizer factory - A Tensorflow version with epsilon under square root """

    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

    def instantiate(self, model: Model) -> RMSpropTF:
        return RMSpropTF(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, alpha=self.alpha, eps=self.eps,
            weight_decay=self.weight_decay, momentum=self.momentum, centered=self.centered
        )


def create(lr, alpha, momentum=0, weight_decay=0, epsilon=1e-8):
    """ Vel factory function """
    return RMSpropTFFactory(lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay, eps=float(epsilon))
