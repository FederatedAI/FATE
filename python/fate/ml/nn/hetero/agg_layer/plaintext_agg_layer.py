import torch
import torch as t
from fate.arch import Context
from typing import Union, List
from torch.autograd import grad
from fate.ml.nn.hetero.agg_layer._base import InteractiveLayer, backward_loss


class InteractiveLayerGuest(InteractiveLayer):

    def __init__(self, ctx: Context,
                 out_features: int,
                 guest_in_features: int,
                 host_in_features: Union[int, List[int]],
                 activation: str = "relu",
                 lr=0.01
                 ):
        super(InteractiveLayerGuest, self).__init__(ctx)
        self._out_features = out_features
        self._guest_in_features = guest_in_features

        assert activation in ["relu", "sigmoid", "tanh"], "activation should be relu, sigmoid or tanh"
        assert isinstance(guest_in_features, int), "guest_in_features should be int"
        assert isinstance(host_in_features, (int, list)), "host_in_features should be int or list[int]"

        self._guest_model = t.nn.Linear(guest_in_features, out_features)
        self._host_model = t.nn.ModuleList()
        if isinstance(host_in_features, int):
            host_in_features = [host_in_features]
        for host_in_feature in host_in_features:
            self._host_model.append(t.nn.Linear(host_in_feature, out_features))

        if activation == "relu":
            self._activation_layer = t.nn.ReLU()
        elif activation == "sigmoid":
            self._activation_layer = t.nn.Sigmoid()
        elif activation == "tanh":
            self._activation_layer = t.nn.Tanh()

        self._host_num = len(self._host_model)
        self._guest_input_cache = None
        self._host_input_caches = None

        self._lr = lr

    def _clear_state(self):
        self._guest_input_cache = None
        self._host_input_caches = None

    def _forward(self, x_g: t.Tensor, x_h: List[t.Tensor]) -> t.Tensor:
        guest_out = self._guest_model(x_g)
        for h_idx in range(self._host_num):
            host_out = self._host_model[h_idx](x_h[h_idx])
            guest_out += host_out
        final_out = self._activation_layer(guest_out)
        return final_out

    def forward(self, x: t.Tensor) -> t.Tensor:

        # save input for backwards
        self._guest_input_cache = t.Tensor(x.detach()).type(t.float64)
        self._host_input_caches = []
        host_x = self.ctx.hosts.get(self._fw_suffix.format(self._fw_count))
        self._fw_count += 1
        for h in range(self._host_num):
            host_input_cache = t.Tensor(host_x[h].detach()).type(t.float64)
            self._host_input_caches.append(host_input_cache)
        with torch.no_grad():
            out = self._forward(self._guest_input_cache, self._host_input_caches)
            final_out = self._activation_layer(out)
            return final_out.detach()

    def backward(self, error) -> t.Tensor:

        # compute backward grads
        self._guest_input_cache = self._guest_input_cache.requires_grad_(True)
        self._host_input_caches = [h.requires_grad_(True) for h in self._host_input_caches]
        out = self._forward(self._guest_input_cache, self._host_input_caches)
        loss = backward_loss(out, error)
        backward_list = [self._guest_input_cache]
        backward_list.extend(self._host_input_caches)
        ret_error = grad(loss, backward_list)

        # update model
        self._guest_input_cache = self._guest_input_cache.requires_grad_(False)
        self._host_input_caches = [h.requires_grad_(False) for h in self._host_input_caches]
        out = self._forward(self._guest_input_cache, self._host_input_caches)
        loss = backward_loss(out, error)
        loss.backward()

        self._clear_state()

        # send error back to hosts
        host_errors = ret_error[1: ]
        idx = 0
        for host in self.ctx.hosts:
            host.put(self._bw_suffix.format(self._bw_count), host_errors[idx])
            idx += 1
        self._bw_count += 1
        return ret_error[0]  # guest error

    def predict(self, x):

        # save input for backwards
        host_x = self.ctx.hosts.get(self._pred_suffix.format(self._pred_count))
        self._pred_count += 1
        with torch.no_grad():
            out = self._forward(x, host_x)
            final_out = self._activation_layer(out)
            return final_out.detach()


class InteractiveLayerHost(InteractiveLayer):

    def __init__(self, ctx):
        super(InteractiveLayerHost, self).__init__(ctx)

    def forward(self, x) -> None:
        self.ctx.guest.put(self._fw_suffix.format(self._fw_count), x)
        self._fw_count += 1
    def backward(self, error=None) -> t.Tensor:
        error = self.ctx.guest.get(self._bw_suffix.format(self._bw_count))
        self._bw_count += 1
        return error

    def predict(self, x):
        self.ctx.guest.put(self._pred_suffix.format(self._pred_count), x)
        self._pred_count += 1
