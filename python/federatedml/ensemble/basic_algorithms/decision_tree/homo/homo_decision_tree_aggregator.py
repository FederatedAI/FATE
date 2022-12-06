from typing import List, Dict
from federatedml.util import LOGGER
from federatedml.framework.weights import DictWeights
from federatedml.framework.homo.aggregator.secure_aggregator import SecureAggregatorClient, SecureAggregatorServer
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_histogram import HistogramBag, \
    FeatureHistogramWeights


class DecisionTreeArbiterAggregator(object):
    """
     secure aggregator for secureboosting Arbiter, gather histogram and numbers
    """

    def __init__(self, verbose=False):
        self.aggregator = SecureAggregatorServer(secure_aggregate=True, communicate_match_suffix='tree_agg')
        self.verbose = verbose

    def aggregate_histogram(self, suffix) -> List[HistogramBag]:

        agg_histogram = self.aggregator.aggregate_model(suffix=suffix)

        if self.verbose:
            for hist in agg_histogram._weights:
                LOGGER.debug('showing aggregated hist{}, hid is {}'.format(hist, hist.hid))

        return agg_histogram._weights

    def aggregate_root_node_info(self, suffix):

        agg_data = self.aggregator.aggregate_model(suffix)
        d = agg_data._weights

        return d['g_sum'], d['h_sum']

    def broadcast_root_info(self, g_sum, h_sum, suffix):

        d = {'g_sum': g_sum, 'h_sum': h_sum}
        self.aggregator.broadcast_model(d, suffix=suffix)


class DecisionTreeClientAggregator(object):
    """
    secure aggregator for secureboosting Client, send histogram and numbers
    """

    def __init__(self, verbose=False):
        self.aggregator = SecureAggregatorClient(
            secure_aggregate=True,
            aggregate_type='sum',
            communicate_match_suffix='tree_agg')
        self.verbose = verbose

    def send_histogram(self, hist: List[HistogramBag], suffix):
        if self.verbose:
            for idx, histbag in enumerate(hist):
                LOGGER.debug('showing client hist {}'.format(histbag))
        weights = FeatureHistogramWeights(list_of_histogram_bags=hist)
        self.aggregator.send_model(weights, suffix=suffix)

    def get_aggregated_root_info(self, suffix) -> Dict:
        gh_dict = self.aggregator.get_aggregated_model(suffix=suffix)
        return gh_dict

    def send_local_root_node_info(self, g_sum, h_sum, suffix):
        d = {'g_sum': g_sum, 'h_sum': h_sum}
        dict_weights = DictWeights(d=d)
        self.aggregator.send_model(dict_weights, suffix=suffix)
