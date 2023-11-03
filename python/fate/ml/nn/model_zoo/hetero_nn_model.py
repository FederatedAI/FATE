import torch
import torch as t
from fate.arch import Context
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerGuest, AggLayerHost


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class HeteroNNModelBase(t.nn.Module):
    
    def __init__(self):
        super().__init__()
        self._bottom_model = None
        self._top_model = None
        self._agg_layer = None
        self._ctx = None
        
    def set_context(self, ctx: Context):
        self._ctx = ctx
        self._agg_layer.set_context(ctx)



class HeteroNNModelGuest(HeteroNNModelBase):

    def __init__(self,
                 top_model: t.nn.Module,
                 agg_layer: AggLayerGuest,
                 bottom_model: t.nn.Module = None
                 ):

        super(HeteroNNModelGuest, self).__init__()
        # cached variables
        if top_model is None:
            raise RuntimeError('guest needs a top model to compute loss, but no top model provided')
        assert isinstance(top_model, t.nn.Module), "top model should be a torch nn.Module"
        self._top_model = top_model
        assert isinstance(agg_layer, AggLayerGuest), "aggregate layer should be a AggLayerGuest"
        self._agg_layer = agg_layer
        if bottom_model is not None:
            assert isinstance(bottom_model, t.nn.Module), "bottom model should be a torch nn.Module"
            self._bottom_model = bottom_model

        self._bottom_fw = None  # for backward usage
        self._agg_fw_rg = None # for backward usage

        # ctx
        self._ctx = None

        # internal mode
        self._guest_direct_backward = True

    def __repr__(self):
        return (f"HeteroNNGuest(top_model={self._top_model}\n"
                f"agg_layer={self._agg_layer}\n"
                f"bottom_model={self._bottom_model})")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw = None
        self._agg_fw_rg = None

    def forward(self, x = None):

        if self._bottom_model is None:
            b_out = None
        else:
            b_out = self._bottom_model(x)
            # bottom layer
            if not self._guest_direct_backward:
                self._bottom_fw = b_out

        # hetero layer
        if not self._guest_direct_backward:
            agg_out = self._agg_layer.forward(b_out)
            self._agg_fw_rg = agg_out.requires_grad_(True)
            # top layer
            top_out = self._top_model(self._agg_fw_rg)
        else:
            top_out = self._top_model(self._agg_layer(b_out))

        return top_out

    def backward(self, loss):

        if self._guest_direct_backward:
            # send error to hosts & guest side direct backward
            self._agg_layer.backward(loss)
            loss.backward()
        else:
            # backward are split into parts
            loss.backward()  # update top
            agg_error = self._agg_fw_rg.grad
            self._agg_fw_rg = None
            bottom_error = self._agg_layer.backward(agg_error)  # compute bottom error & update hetero
            if bottom_error is not None:
                bottom_loss = backward_loss(self._bottom_fw, bottom_error)
                bottom_loss.backward()
            self._bottom_fw = False

    def predict(self, x = None):

        with torch.no_grad():
            if self._bottom_model is None:
                b_out = None
            else:
                b_out = self._bottom_model(x)
            agg_out = self._agg_layer.predict(b_out)
            top_out = self._top_model(agg_out)

        return top_out


class HeteroNNModelHost(HeteroNNModelBase):

    def __init__(self,
                 agg_layer: AggLayerHost,
                 bottom_model: t.nn.Module
                 ):

        super().__init__()

        assert isinstance(bottom_model, t.nn.Module), "bottom model should be a torch nn.Module"
        self._bottom_model = bottom_model
        assert isinstance(agg_layer, AggLayerHost), "aggregate layer should be a AggLayerHost"
        self._agg_layer = agg_layer
        # cached variables
        self._bottom_fw = None  # for backward usage
        # ctx
        self._ctx = None

    def __repr__(self):
        return f"HeteroNNHost(bottom_model={self._bottom_model})"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw = None

    def set_context(self, ctx: Context):
        self._ctx = ctx
        self._agg_layer.set_context(ctx)

    def forward(self, x):

        b_out = self._bottom_model(x)
        # bottom layer
        self._bottom_fw = b_out
        # hetero layer
        self._agg_layer.forward(b_out)

    def backward(self):

        error = self._agg_layer.backward()
        loss = backward_loss(self._bottom_fw, error)
        loss.backward()
        self._clear_state()

    def predict(self, x):

        with torch.no_grad():
            b_out = self._bottom_model(x)
            self._agg_layer.predict(b_out)