#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
from typing import List, Literal
import torch as t
from fate.arch import Context
from torch.nn.modules.module import T

MERGE_TYPE = ["sum", "concat"]


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class _AggLayerBase(t.nn.Module):
    def __init__(self):
        super().__init__()
        self._ctx = None
        self._fw_suffix = "agglayer_fw_{}"
        self._bw_suffix = "agglayer_bw_{}"
        self._pred_suffix = "agglayer_pred_{}"
        self._fw_count = 0
        self._bw_count = 0
        self._pred_count = 0
        self._has_ctx = False
        self._model = None
        self.training = True
        self.device = None

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

    def set_device(self, device):
        self.device = device

    @property
    def ctx(self):
        if self._ctx is None or self._has_ctx == False:
            raise ValueError("Context is not set yet, please call set_context() first")
        return self._ctx

    def _clear_state(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self: T, mode: bool = True) -> T:
        self.training = mode
        return self

    def eval(self: T) -> T:
        self.training = False
        return self


class AggLayerGuest(_AggLayerBase):
    def __init__(self, merge_type: Literal["sum", "concat"] = "sum", concat_dim=1):
        super(AggLayerGuest, self).__init__()
        self._host_input_caches = None
        self._merge_type = merge_type
        assert self._merge_type in MERGE_TYPE, f"merge type should be one of {MERGE_TYPE}"
        assert isinstance(concat_dim, int), "concat dim should be int"
        self.register_buffer("_concat_dim", torch.LongTensor(concat_dim))

    def _clear_state(self):
        self._host_input_caches = None

    def _forward(self, x_g: t.Tensor = None, x_h: List[t.Tensor] = None) -> t.Tensor:
        if x_g is None and x_h is None:
            raise ValueError("guest input and host inputs cannot be both None")

        if x_g is not None:
            x_g = x_g.to(self.device)
        if x_h is not None:
            x_h = [h.to(self.device) for h in x_h]

        can_cat = True
        if x_g is None:
            x_g = 0
            can_cat = False
        else:
            if self._model is not None:
                x_g = self._model(x_g)

        if x_h is None:
            ret = x_g
        else:
            if self._merge_type == "sum":
                for h_idx in range(len(x_h)):
                    x_g += x_h[h_idx]
                ret = x_g
            elif self._merge_type == "concat":
                # xg + x_h
                feat = [x_g] if can_cat else []
                feat.extend(x_h)
                ret = torch.cat(feat, dim=1)
            else:
                raise RuntimeError("unknown merge type")

        return ret

    def _get_fw_from_host(self):
        host_x = self.ctx.hosts.get(self._fw_suffix.format(self._fw_count))
        self._fw_count += 1
        return host_x

    def _send_err_to_host(self, ret_error):
        host_errors = ret_error
        idx = 0
        for host in self.ctx.hosts:
            host.put(self._bw_suffix.format(self._bw_count), host_errors[idx])
            idx += 1
        self._bw_count += 1

    def forward(self, x: t.Tensor = None) -> t.Tensor:
        if self.training:
            if self._has_ctx:
                self._host_input_caches = []
                host_x = self._get_fw_from_host()
                for h in range(len(host_x)):
                    host_input_cache = t.from_numpy(host_x[h]).requires_grad_(True)
                    self._host_input_caches.append(host_input_cache)
            else:
                self._host_input_caches = None

            final_out = self._forward(x, self._host_input_caches)
            return final_out

        else:
            return self.predict(x)

    def backward(self, error):
        # compute backward grads
        backward_list = []
        if self._host_input_caches is not None and self._has_ctx:
            for h in self._host_input_caches:
                backward_list.append(h)
            ret_error = t.autograd.grad(error, backward_list, retain_graph=True)
            # send error back to hosts
            self._send_err_to_host(ret_error)
            self._clear_state()

    def predict(self, x):
        host_x = None
        if self._has_ctx:
            host_x = self.ctx.hosts.get(self._pred_suffix.format(self._pred_count))
            self._pred_count += 1
            host_x = [t.from_numpy(h) for h in host_x]
        with torch.no_grad():
            out = self._forward(x, host_x)
            return out


class AggLayerHost(_AggLayerBase):
    def __init__(self):
        super(AggLayerHost, self).__init__()
        self._out_cache = None
        self._input_cache = None

    def _send_fw_to_guest(self, x):
        self.ctx.guest.put(self._fw_suffix.format(self._fw_count), x)
        self._fw_count += 1

    def _get_error_from_guest(self):
        error = self.ctx.guest.get(self._bw_suffix.format(self._bw_count))
        self._bw_count += 1
        return error

    def _clear_state(self):
        self._out_cache, self._host_input_caches = None, None

    def forward(self, x: t.Tensor) -> None:
        if self.training:
            assert isinstance(x, t.Tensor), "x should be a tensor"
            if self._model is not None:
                self._input_cache = t.from_numpy(x.cpu().detach().numpy()).to(self.device).requires_grad_(True)
                out_ = self._model(self._input_cache)
                self._out_cache = out_
            else:
                out_ = x
            self._send_fw_to_guest(out_.detach().cpu().numpy())
        else:
            self.predict(x)

    def backward(self, error=None) -> t.Tensor:
        error = self._get_error_from_guest()
        if self._input_cache is not None and self._model is not None:
            error = error.to(self.device)
            loss = backward_loss(self._out_cache, error)
            loss.backward()
            error = self._input_cache.grad
            self._clear_state()
            return error
        else:
            return error

    def predict(self, x):
        with torch.no_grad():
            if self._model is not None:
                out_ = self._model(x)
            else:
                out_ = x
            self.ctx.guest.put(self._pred_suffix.format(self._pred_count), out_.detach().cpu().numpy())
            self._pred_count += 1
