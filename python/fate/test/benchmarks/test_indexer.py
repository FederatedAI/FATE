import fate_utils

nids = [2] * 1000
bids = [[20] * 1000] * 100
indexer = fate_utils.histogram.HistogramIndexer(5, [34] * 100)


def loop_get_position(indexer, nids, b):
    for fid, bids in enumerate(b):
        for nid, bid in zip(nids, bids):
            indexer.get_position(nid, fid, bid)


def test_rust_loop_get_position(benchmark):
    benchmark(lambda: loop_get_position(indexer, nids, bids))


def test_rust_get_positions(benchmark):
    benchmark(lambda: indexer.get_positions(nids, bids))
