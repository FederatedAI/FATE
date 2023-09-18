from fate.arch import Context
from fate.ml.aggregator.base import BaseAggregatorClient, BaseAggregatorServer


class SecureAggregatorClient(BaseAggregatorClient):

    def __init__(self, ctx: Context, aggregator_name: str = None, aggregate_type='mean', sample_num=1) -> None:
        super().__init__(ctx, aggregator_name, aggregate_type, sample_num, is_mock=False)


class SecureAggregatorServer(BaseAggregatorServer):

    def __init__(self, ctx: Context, aggregator_name: str = None) -> None:
        super().__init__(ctx, aggregator_name, is_mock=False)
