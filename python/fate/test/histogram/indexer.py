import pytest

# from fate.arch.histogram.indexer import HistogramIndexer, Shuffler
from fate_utils.histogram import HistogramIndexer, Shuffler


class TestHistogramIndexer:
    @pytest.fixture
    def histogram_indexer(self):
        node_size = 2
        feature_bin_sizes = [3, 2]
        return HistogramIndexer(node_size, feature_bin_sizes)

    def test_get_position(self, histogram_indexer):
        assert histogram_indexer.get_position(0, 0, 0) == 0
        assert histogram_indexer.get_position(0, 0, 1) == 1
        assert histogram_indexer.get_position(0, 1, 0) == 3
        assert histogram_indexer.get_position(1, 0, 0) == 5
        assert histogram_indexer.get_position(1, 1, 1) == 9

    def test_get_reverse_position(self, histogram_indexer):
        assert histogram_indexer.get_reverse_position(0) == (0, 0, 0)
        assert histogram_indexer.get_reverse_position(1) == (0, 0, 1)
        assert histogram_indexer.get_reverse_position(3) == (0, 1, 0)
        assert histogram_indexer.get_reverse_position(5) == (1, 0, 0)
        assert histogram_indexer.get_reverse_position(9) == (1, 1, 1)

    def test_get_node_intervals(self, histogram_indexer):
        assert histogram_indexer.get_node_intervals() == [(0, 5), (5, 10)]

    def test_get_feature_position_ranges(self, histogram_indexer):
        assert histogram_indexer.get_feature_position_ranges() == [(0, 3), (3, 5), (5, 8), (8, 10)]

    def test_total_data_size(self, histogram_indexer):
        assert histogram_indexer.total_data_size() == 10

    def test_splits_into_k(self, histogram_indexer):
        # Test if the method splits the data correctly into k parts
        k = 2
        splits = list(histogram_indexer.splits_into_k(k))

        assert len(splits) == k

        # Check if the intervals are disjoint and cover the entire range
        all_intervals = [interval for _, _, intervals in splits for interval in intervals]
        all_intervals.sort(key=lambda x: x[0])

        assert all_intervals[0][0] == 0
        assert all_intervals[-1][1] == histogram_indexer.total_data_size()
        for i in range(len(all_intervals) - 1):
            assert all_intervals[i][1] == all_intervals[i + 1][0]

    def test_one_node_data_size(self, histogram_indexer):
        # Test if the one node data size is correctly calculated
        assert histogram_indexer.one_node_data_size() == sum(histogram_indexer.feature_bin_sizes)

    def test_global_flatten_bin_sizes(self, histogram_indexer):
        # Test if the global flatten bin sizes is correctly calculated
        assert (
            histogram_indexer.global_flatten_bin_sizes()
            == histogram_indexer.feature_bin_sizes * histogram_indexer.node_size
        )

    def test_flatten_in_node(self, histogram_indexer):
        # Test if the flatten in node method returns a new HistogramIndexer with correct parameters
        new_indexer = histogram_indexer.flatten_in_node()

        assert new_indexer.node_size == histogram_indexer.node_size
        assert new_indexer.feature_bin_sizes == [histogram_indexer.one_node_data_size()]

    def test_squeeze_bins(self, histogram_indexer):
        # Test if the squeeze bins method returns a new HistogramIndexer with correct parameters
        new_indexer = histogram_indexer.squeeze_bins()

        assert new_indexer.node_size == histogram_indexer.node_size
        assert new_indexer.feature_bin_sizes == [1] * histogram_indexer.feature_size

    def test_reshape(self, histogram_indexer):
        # Test if the reshape method returns a new HistogramIndexer with correct parameters
        new_feature_bin_sizes = [2, 2, 1]
        new_indexer = histogram_indexer.reshape(new_feature_bin_sizes)

        assert new_indexer.node_size == histogram_indexer.node_size
        assert new_indexer.feature_bin_sizes == new_feature_bin_sizes

    # def test_get_shuffler(self, histogram_indexer):
    #     # Test if the get shuffler method returns a Shuffler with correct parameters
    #     seed = 123
    #     shuffler = histogram_indexer.get_shuffler(seed)
    #
    #     assert isinstance(shuffler, Shuffler)
    #     assert shuffler.num_node == histogram_indexer.node_size
    #     assert shuffler.node_size == histogram_indexer.one_node_data_size()

    def test_unflatten_indexes(self, histogram_indexer):
        # Test if the unflatten indexes method returns a correct nested dictionary of indexes
        indexes = histogram_indexer.unflatten_indexes()

        for nid in range(histogram_indexer.node_size):
            for fid in range(histogram_indexer.feature_size):
                for bid in range(histogram_indexer.feature_bin_sizes[fid]):
                    assert indexes[nid][fid][bid] == histogram_indexer.get_position(nid, fid, bid)
