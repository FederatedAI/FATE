import torch
import torch as t
from fate.arch import Context
from fate.ml.nn.hetero.agg_layer.plaintext_agg_layer import InteractiveLayerHost


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class SplitNNHost(t.nn.Module):

    def __init__(self, bottom_model: t.nn.Module,
                 interactive_layer: InteractiveLayerHost,
                 ctx: Context
                 ):

        super().__init__()
        self._bottom_model = bottom_model
        self._interactive_layer = interactive_layer

        # cached variables
        self._bottom_fw_rg = None  # for backward usage
        self._bottom_fw = None  # for forward & error compute

        # ctx
        self._ctx = ctx

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw_rg = None
        self._bottom_fw = None

    def forward(self, x):

        b_out = self._bottom_model(x)
        # bottom layer
        self._bottom_fw_rg = b_out
        self._bottom_fw = t.Tensor(b_out.detach()).requires_grad_(False)
        # hetero layer
        self._interactive_layer.forward(b_out)

    def backward(self):

        error = self._interactive_layer.backward()
        loss = backward_loss(self._bottom_fw_rg, error)
        loss.backward()
        self._clear_state()

    def predict(self, x):

        with torch.no_grad():
            b_out = self._bottom_model(x)
            self._interactive_layer.predict(b_out)