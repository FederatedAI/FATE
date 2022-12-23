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

    def sum(self, x, *args, **kwargs):
        return self.apply_signature1("sum", args, kwargs)(x)

    def sqrt(self, x, *args, **kwargs):
        return self.apply_signature1("sqrt", args, kwargs)(x)

    def add(self, x, y, *args, **kwargs):
        return self.apply_signature2("add", args, kwargs)(x, y)

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
