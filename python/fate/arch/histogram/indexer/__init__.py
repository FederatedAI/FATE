INDEXER_USE_PYTHON = True

if INDEXER_USE_PYTHON:
    from ._indexer import HistogramIndexer, Shuffler
# else:
#     from fate_utils.histogram import HistogramIndexer, Shuffler

__all__ = ["HistogramIndexer", "Shuffler"]
