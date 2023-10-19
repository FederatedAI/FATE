import torch
import torch as t
from torch.autograd import grad
from fate.arch import Context
from fate.ml.nn.hetero.agg_layer.plaintext_agg_layer import InteractiveLayerGuest


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class SplitNNGuest(t.nn.Module):

    def __init__(self,
                 ctx: Context,
                 bottom_model: t.nn.Module,
                 top_model: t.nn.Module,
                 interactive_layer: InteractiveLayerGuest,
                 ):

        super(SplitNNGuest, self).__init__()
        self._bottom_model = bottom_model
        self._top_model = top_model
        self._interactive_layer = interactive_layer

        # cached variables
        self._bottom_fw_rg = None  # for backward usage
        self._bottom_fw = None  # for forward & error compute
        self._interactive_fw_rg = None # for backward usage
        self._interactive_fw = None # for fw & error compute

        # ctx
        self._ctx = ctx

    def __repr__(self):
        return f"HeteroNNGuest(bottom_model={self._bottom_model}\ntop_model={self._top_model})"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw_rg = None
        self._bottom_fw = None
        self._interactive_fw_rg = None
        self._interactive_fw = None
    def forward(self, x):

        b_out = self._bottom_model(x)
        # bottom layer
        self._bottom_fw_rg = b_out
        self._bottom_fw = t.Tensor(b_out.detach()).requires_grad_(False)
        # hetero layer
        interactive_out = self._interactive_layer.forward(b_out)
        self._interactive_fw_rg = interactive_out.requires_grad_(True)
        self._interactive_fw = interactive_out
        # top layer
        top_out = self._top_model(self._interactive_fw_rg)

        return top_out

    def backward(self, loss):

        interactive_error = grad(loss, self._interactive_fw_rg, retain_graph=True)[0]  # compute backward error
        loss.backward()  # update top
        bottom_error = self._interactive_layer.backward(interactive_error)  # compute bottom error & update hetero
        bottom_loss = backward_loss(self._bottom_fw_rg, bottom_error)
        bottom_loss.backward()
        self._clear_state()

    def predict(self, x):

        with torch.no_grad():
            b_out = self._bottom_model(x)
            interactive_out = self._interactive_layer.predict(b_out)
            top_out = self._top_model(interactive_out)

        return top_out
