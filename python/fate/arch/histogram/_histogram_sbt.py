from ._histogram_distributed import DistributedHistogram
from ._histogram_local import Histogram


class HistogramBuilder:
    def __init__(
        self,
        num_node,
        feature_bin_sizes,
        value_schemas,
        global_seed=None,
        seed=None,
        node_mapping=None,
        k=None,
        enable_cumsum=True,
    ):
        self._num_node = num_node
        self._feature_bin_sizes = feature_bin_sizes
        self._node_data_size = sum(feature_bin_sizes)
        self._value_schemas = value_schemas
        self._global_seed = global_seed
        self._seed = seed
        self._node_mapping = node_mapping
        self._enable_cumsum = enable_cumsum
        self._k = k

    def __str__(self):
        return f"<{self.__class__.__name__} node_size={self._num_node}, feature_bin_sizes={self._feature_bin_sizes}, node_data_size={self._node_data_size}, seed={self._global_seed}>"

    def statistic(self, data) -> "DistributedHistogram":
        """
        Update the histogram with the data.
        Args:
            data: table with the following schema:
        Returns:
            ShuffledHistogram, the shuffled histogram
        """
        if self._k is None:
            self._k = data.partitions**2
        mapper = get_partition_hist_build_mapper(
            self._num_node,
            self._feature_bin_sizes,
            self._value_schemas,
            self._global_seed,
            self._k,
            self._node_mapping,
            self._enable_cumsum,
        )
        table = data.mapReducePartitions(mapper, lambda x, y: x.iadd(y))
        data = DistributedHistogram(
            table, self._k, self._num_node, self._node_data_size, global_seed=self._global_seed, seed=self._seed
        )
        return data


def get_partition_hist_build_mapper(
    num_node, feature_bin_sizes, value_schemas, global_seed, k, node_mapping, enable_cumsum
):
    def _partition_hist_build_mapper(part):
        hist = Histogram.create(num_node, feature_bin_sizes, value_schemas)
        for _, raw in part:
            feature_ids, node_ids, targets = raw
            hist.i_update(feature_ids, node_ids, targets, node_mapping)
        if enable_cumsum:
            hist.i_cumsum_bins()
        if global_seed is not None:
            hist.i_shuffle(global_seed)
        splits = hist.to_splits(k)
        return splits

    return _partition_hist_build_mapper
