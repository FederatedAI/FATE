#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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


def initialize_param(coef_len, **kwargs):
    param_len = coef_len
    method = kwargs["method"]
    fit_intercept = kwargs["fit_intercept"]
    random_state = kwargs.get("random_state", None)
    if fit_intercept:
        param_len = param_len + 1
    if method == 'zeros':
        return torch.zeros((param_len, 1), requires_grad=True)
    elif method == 'ones':
        return torch.ones((param_len, 1), requires_grad=True)
    elif method == 'consts':
        return torch.full(
            (param_len, 1), float(
                kwargs["fill_val"]), requires_grad=True)
    elif method == 'random':
        if random_state is not None:
            generator = torch.Generator().manual_seed(random_state)
            return torch.randn((param_len, 1), generator=generator, requires_grad=True)
        return torch.randn((param_len, 1), requires_grad=True)
    elif method == 'random_uniform':
        if random_state is not None:
            generator = torch.Generator().manual_seed(random_state)
            return torch.rand((param_len, 1), generator=generator, requires_grad=True)
        return torch.rand((param_len, 1), requires_grad=True)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")


def serialize_param(param, fit_intercept=False):
    dtype = str(param.dtype).split(".", -1)[-1]
    w = param.tolist()
    intercept = None
    if fit_intercept:
        intercept = w[-1]
        w = w[:-1]
    return {"coef_": w, "intercept_": intercept, "dtype": dtype}


def deserialize_param(param, fit_intercept=False):
    w = param["coef_"]
    if fit_intercept:
        w.append(param["intercept_"])
    dtype = param["dtype"]
    w = torch.tensor(w, dtype=getattr(torch, dtype))
    return w


def check_overflow(param, threshold=1e8):
    if (torch.abs(param) > threshold).any():
        raise ValueError(f"Value(s) greater than {threshold} found in model param, please check.")
