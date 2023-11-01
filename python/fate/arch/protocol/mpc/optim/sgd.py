#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from .optimizer import Optimizer
from ..cryptensor import CrypTensor


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        grad_threshold (float, optional): imposes a threshold on the magnitude of gradient values.
            Gradient values with magnitude above the threshold will be replaced with 0.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        grad_threshold=None,
    ):
        if not isinstance(lr, (int, float)) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(momentum, (int, float)) or momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not isinstance(dampening, (int, float)):
            raise ValueError("Invalid dampening value {}".format(dampening))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Compute thresholding based on square value since abs is more expensive
        self.square_threshold = grad_threshold
        if self.square_threshold is not None:
            self.square_threshold *= self.square_threshold

        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with CrypTensor.no_grad():
            loss = None
            if closure is not None:
                with CrypTensor.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    # Threshold gradients to prevent gradient explosion
                    if self.square_threshold is not None:
                        d_p = p.grad.mul(p.grad.square().lt(self.square_threshold))
                    else:
                        d_p = p.grad

                    if weight_decay != 0:
                        d_p = d_p.add(p.mul(weight_decay))
                    if momentum != 0:
                        param_state = self.state[id(p)]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone().detach()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p.mul(1 - dampening))
                        if nesterov:
                            d_p = d_p.add(buf.mul(momentum))
                        else:
                            d_p = buf

                    p.sub_(d_p.mul(group["lr"]))

            return loss
