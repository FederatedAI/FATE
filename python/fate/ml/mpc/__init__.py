from fate.arch.context import Context


class MPCModule:
    @classmethod
    def parse_parameters(cls, parameters):
        import inspect

        for sig_name, sig in inspect.signature(cls.__init__).parameters.items():
            if sig_name in parameters:
                if sig.annotation is not sig.empty:
                    parameters[sig_name] = sig.annotation(parameters[sig_name])
                elif sig.default is not sig.empty:
                    parameters[sig_name] = type(sig.default)(parameters[sig_name])
        return parameters

    def fit(
        self,
        ctx: Context,
    ) -> None:
        ...
