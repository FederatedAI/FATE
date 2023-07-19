from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient, PlainTextAggregatorServer
from fate.ml.aggregator.secure_aggregator import SecureAggregatorClient, SecureAggregatorServer
import enum


class AggregatorType(enum.Enum):
    PLAINTEXT = 'plaintext'
    SECURE_AGGREGATE = 'secure_aggregate'


aggregator_map = {
    AggregatorType.PLAINTEXT.value: (PlainTextAggregatorClient, PlainTextAggregatorServer),
    AggregatorType.SECURE_AGGREGATE.value: (SecureAggregatorClient, SecureAggregatorServer)
}


__all__ = ['PlainTextAggregatorClient', 'PlainTextAggregatorServer', 'SecureAggregatorClient', 'SecureAggregatorServer']
