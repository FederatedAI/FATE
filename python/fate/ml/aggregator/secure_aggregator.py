from fate.arch import Context


class SecureAggregatorClient(object):

    def __init__(self, ctx: Context, aggregate_type='weighted_mean', aggregate_weight=1.0) -> None:
        self.aggregate_type = aggregate_type
        self.aggregate_weight = aggregate_weight
        self.ctx = ctx

    def aggregate(self):
        pass


class SecureAggregatorServer(object):

    def __init__(self, ctx: Context) -> None:
        pass

    def aggregate(self):
        pass
    
