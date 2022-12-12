from torch import optim
from pipeline.component.nn.backend.torch.base import FateTorchOptimizer


class ASGD(optim.ASGD, FateTorchOptimizer):

    def __init__(
        self,
        params=None,
        lr=0.01,
        lambd=0.0001,
        alpha=0.75,
        t0=1000000.0,
        weight_decay=0,
        foreach=None,
        maximize=False,
    ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['lambd'] = lambd
        self.param_dict['alpha'] = alpha
        self.param_dict['t0'] = t0
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['foreach'] = foreach
        self.param_dict['maximize'] = maximize
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.ASGD.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer ASGD without initiated parameters'.format(type(self).__name__)


class Adadelta(optim.Adadelta, FateTorchOptimizer):

    def __init__(self, params=None, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['rho'] = rho
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['foreach'] = foreach
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.Adadelta.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer Adadelta without initiated parameters'.format(type(self).__name__)


class Adagrad(optim.Adagrad, FateTorchOptimizer):

    def __init__(
        self,
        params=None,
        lr=0.01,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach=None,
    ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['lr_decay'] = lr_decay
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['initial_accumulator_value'] = initial_accumulator_value
        self.param_dict['eps'] = eps
        self.param_dict['foreach'] = foreach
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.Adagrad.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer Adagrad without initiated parameters'.format(type(self).__name__)


class Adam(optim.Adam, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['amsgrad'] = amsgrad
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.Adam.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer Adam without initiated parameters'.format(type(self).__name__)


class AdamW(optim.AdamW, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['amsgrad'] = amsgrad
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.AdamW.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer AdamW without initiated parameters'.format(type(self).__name__)


class Adamax(optim.Adamax, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['foreach'] = foreach
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.Adamax.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer Adamax without initiated parameters'.format(type(self).__name__)


class LBFGS(optim.LBFGS, FateTorchOptimizer):

    def __init__(
        self,
        params=None,
        lr=1,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-07,
        tolerance_change=1e-09,
        history_size=100,
        line_search_fn=None,
    ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['max_iter'] = max_iter
        self.param_dict['max_eval'] = max_eval
        self.param_dict['tolerance_grad'] = tolerance_grad
        self.param_dict['tolerance_change'] = tolerance_change
        self.param_dict['history_size'] = history_size
        self.param_dict['line_search_fn'] = line_search_fn
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.LBFGS.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer LBFGS without initiated parameters'.format(type(self).__name__)


class NAdam(optim.NAdam, FateTorchOptimizer):

    def __init__(
        self,
        params=None,
        lr=0.002,
        betas=(
            0.9,
            0.999),
        eps=1e-08,
        weight_decay=0,
        momentum_decay=0.004,
        foreach=None,
    ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['momentum_decay'] = momentum_decay
        self.param_dict['foreach'] = foreach
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.NAdam.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer NAdam without initiated parameters'.format(type(self).__name__)


class RAdam(optim.RAdam, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['foreach'] = foreach
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.RAdam.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer RAdam without initiated parameters'.format(type(self).__name__)


class RMSprop(optim.RMSprop, FateTorchOptimizer):

    def __init__(
        self,
        params=None,
        lr=0.01,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
        foreach=None,
        maximize=False,
        differentiable=False,
    ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['alpha'] = alpha
        self.param_dict['eps'] = eps
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['momentum'] = momentum
        self.param_dict['centered'] = centered
        self.param_dict['foreach'] = foreach
        self.param_dict['maximize'] = maximize
        self.param_dict['differentiable'] = differentiable
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.RMSprop.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer RMSprop without initiated parameters'.format(type(self).__name__)


class Rprop(optim.Rprop, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50), foreach=None, maximize=False, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['etas'] = etas
        self.param_dict['step_sizes'] = step_sizes
        self.param_dict['foreach'] = foreach
        self.param_dict['maximize'] = maximize
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.Rprop.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer Rprop without initiated parameters'.format(type(self).__name__)


class SGD(optim.SGD, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['momentum'] = momentum
        self.param_dict['dampening'] = dampening
        self.param_dict['weight_decay'] = weight_decay
        self.param_dict['nesterov'] = nesterov
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.SGD.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer SGD without initiated parameters'.format(type(self).__name__)


class SparseAdam(optim.SparseAdam, FateTorchOptimizer):

    def __init__(self, params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False, ):
        FateTorchOptimizer.__init__(self)
        self.param_dict['lr'] = lr
        self.param_dict['betas'] = betas
        self.param_dict['eps'] = eps
        self.param_dict['maximize'] = maximize
        self.torch_class = type(self).__bases__[0]

        if params is None:
            return

        params = self.check_params(params)

        self.torch_class.__init__(self, params, **self.param_dict)

        # optim.SparseAdam.__init__(self, **self.param_dict)

    def __repr__(self):
        try:
            return type(self).__bases__[0].__repr__(self)
        except BaseException:
            return 'Optimizer SparseAdam without initiated parameters'.format(type(self).__name__)
