import logging
import typing

from ._histogram_splits import HistogramSplits
from .indexer import HistogramIndexer, Shuffler
from .values import HistogramValuesContainer

logger = logging.getLogger(__name__)


class Histogram:
    def __init__(self, indexer: "HistogramIndexer", values: HistogramValuesContainer):
        self._indexer = indexer
        self._data = values

    def __str__(self):
        return self._data.show(self._indexer)

    @classmethod
    def create(cls, num_node, feature_bin_sizes, values_schema: dict):
        indexer = HistogramIndexer(num_node, feature_bin_sizes)
        size = indexer.total_data_size()
        return cls(indexer, HistogramValuesContainer.create(values_schema, size))

    def i_update(self, fids, nids, targets, node_mapping):
        positions = self._indexer.get_positions(
            nids.flatten().detach().numpy().tolist(), fids.detach().numpy().tolist(), node_mapping
        )
        self._data.i_update(targets, positions)
        return self

    def iadd(self, hist: "Histogram"):
        self._data.iadd(hist._data)
        return self

    def decrypt(self, sk_map: dict):
        return Histogram(self._indexer, self._data.decrypt(sk_map))

    def decode(self, coder_map: dict):
        return Histogram(self._indexer, self._data.decode(coder_map))

    def i_shuffle(self, seed, reverse=False):
        shuffler = Shuffler(self._indexer.get_node_size(), self._indexer.get_node_axis_stride(), seed)
        self._data.i_shuffle(shuffler, reverse=reverse)
        return self

    def i_cumsum_bins(self):
        self._data.i_cumsum_bins(self._indexer.global_flatten_bin_sizes())
        return self

    def reshape(self, feature_bin_sizes):
        indexer = self._indexer.reshape(feature_bin_sizes)
        return Histogram(indexer, self._data)

    def extract_data(self):
        return self._data.extract_data(self._indexer)

    def to_splits(self, k) -> typing.Iterator[typing.Tuple[(int, "HistogramSplits")]]:
        for pid, (start, end), indexes in self._indexer.splits_into_k(k):
            data = self._data.intervals_slice(indexes)
            yield pid, HistogramSplits(pid, self._indexer.node_size, start, end, data)
