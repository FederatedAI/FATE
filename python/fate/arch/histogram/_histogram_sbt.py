from ._histogram_distributed import DistributedHistogram
from ._histogram_local import Histogram


class HistogramBuilder:
    def __init__(self, node_size, feature_bin_sizes, value_schemas, seed):
        self._node_size = node_size
        self._feature_bin_sizes = feature_bin_sizes
        self._node_data_size = sum(feature_bin_sizes)
        self._value_schemas = value_schemas
        self._seed = seed

    def __str__(self):
        return f"<{self.__class__.__name__} node_size={self._node_size}, feature_bin_sizes={self._feature_bin_sizes}, node_data_size={self._node_data_size}, seed={self._seed}>"

    def statistic(self, data, k=None, node_mapping=None) -> "DistributedHistogram":
        """
        Update the histogram with the data.
        Args:
            data: table with the following schema:
            k: number of output splits of the histogram
        Returns:
            ShuffledHistogram, the shuffled(if seed is not None) histogram
        """
        if k is None:
            k = data.partitions ** 2
        mapper = get_partition_hist_build_mapper(
            self._node_size, self._feature_bin_sizes, self._value_schemas, self._seed, k, node_mapping,
        )
        table = data.mapReducePartitions(mapper, lambda x, y: x.iadd(y))
        return DistributedHistogram(table, self._node_size, self._node_data_size)


def get_partition_hist_build_mapper(node_size, feature_bin_sizes, value_schemas, seed, k, node_mapping):
    def _partition_hist_build_mapper(part):
        hist = Histogram.create(node_size, feature_bin_sizes, value_schemas)
        for _, raw in part:
            fids, nids, targets = raw
            hist.i_update(fids, nids, targets, node_mapping)
        hist.i_cumsum_bins()
        hist.i_shuffle(seed)
        splits = hist.to_splits(k)
        return splits

    return _partition_hist_build_mapper
