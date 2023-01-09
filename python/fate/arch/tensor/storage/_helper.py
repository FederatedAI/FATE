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
class local_ops_helper:
    def __init__(self, device, dtype) -> None:
        self.device = device
        self.dtype = dtype

    def square(self, x, *args, **kwargs):
        return self.apply_signature1("square", args, kwargs)(x)

    def var(self, x, *args, **kwargs):
        return self.apply_signature1("var", args, kwargs)(x)

    def std(self, x, *args, **kwargs):
        return self.apply_signature1("std", args, kwargs)(x)

    def max(self, x, *args, **kwargs):
        return self.apply_signature1("max", args, kwargs)(x)

    def min(self, x, *args, **kwargs):
        return self.apply_signature1("min", args, kwargs)(x)

    def sum(self, x, *args, **kwargs):
        return self.apply_signature1("sum", args, kwargs)(x)

    def sqrt(self, x, *args, **kwargs):
        return self.apply_signature1("sqrt", args, kwargs)(x)

    def add(self, x, y, *args, **kwargs):
        return self.apply_signature2("add", args, kwargs)(x, y)

    def maximum(self, x, y, *args, **kwargs):
        return self.apply_signature2("maximum", args, kwargs)(x, y)

    def minimum(self, x, y, *args, **kwargs):
        return self.apply_signature2("minimum", args, kwargs)(x, y)

    def div(self, x, y, *args, **kwargs):
        return self.apply_signature2("div", args, kwargs)(x, y)

    def sub(self, x, y, *args, **kwargs):
        return self.apply_signature2("sub", args, kwargs)(x, y)

    def mul(self, x, y, *args, **kwargs):
        return self.apply_signature2("mul", args, kwargs)(x, y)

    def truediv(self, x, y, *args, **kwargs):
        return self.apply_signature2("true_divide", args, kwargs)(x, y)

    def matmul(self, x, y, *args, **kwargs):
        return self.apply_signature2("matmul", args, kwargs)(x, y)

    def slice(self, x, *args, **kwargs):
        return self.apply_signature1("slice", args, kwargs)(x)

    def apply_signature1(self, method, args, kwargs):
        from .local.device import _ops_dispatch_signature1_local_unknown_unknown

        return _ops_dispatch_signature1_local_unknown_unknown(method, self.device, self.dtype, args, kwargs)

    def apply_signature2(self, method, args, kwargs):
        from .local.device import _ops_dispatch_signature2_local_unknown_unknown

        return _ops_dispatch_signature2_local_unknown_unknown(method, self.device, self.dtype, args, kwargs)
