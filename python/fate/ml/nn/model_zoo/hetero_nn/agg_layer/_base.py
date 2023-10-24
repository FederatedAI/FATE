import torch as t
from fate.arch import Context

def backward_loss(z, backward_error):
    return t.sum(z * backward_error)

class AggLayer(t.nn.Module):
    def __init__(self):
        super().__init__()
        self._ctx = None
        self._fw_suffix = "interactive_fw_{}"
        self._bw_suffix = "interactive_bw_{}"
        self._pred_suffix = "interactive_pred_{}"
        self._fw_count = 0
        self._bw_count = 0
        self._pred_count = 0
        self._has_ctx = False

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, error):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def set_context(self, ctx: Context):
        self._ctx = ctx
        self._has_ctx = True
    def has_context(self):
        return self._has_ctx
    @property
    def ctx(self):
        if self._ctx is None or self._has_ctx == False:
            raise ValueError("Context is not set yet, please call set_context() first")
        return self._ctx

    def _clear_state(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


