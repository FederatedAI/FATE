import torch as t
from fate.arch import Context

def backward_loss(z, backward_error):
    return t.sum(z * backward_error)

class InteractiveLayer(t.nn.Module):
    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx
        self._fw_suffix = "interactive_fw_{}"
        self._bw_suffix = "interactive_bw_{}"
        self._pred_suffix = "interactive_pred_{}"
        self._fw_count = 0
        self._bw_count = 0
        self._pred_count = 0

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, error):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def _clear_state(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


