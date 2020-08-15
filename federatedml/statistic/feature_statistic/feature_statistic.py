import functools
from collections import Counter

from federatedml.feature.sparse_vector import SparseVector

from federatedml.model_base import ModelBase
from federatedml.feature.instance import Instance

from arch.api.utils import log_utils

import numpy as np

LOGGER = log_utils.getLogger()


def count_miss_feat(feat_set: set, all_feat: set, important_feat: set = None):

    assert len(all_feat) > 0, 'len of feature list must larger than 0'
    exist_num = len(all_feat.intersection(feat_set))
    lost_perc = 1 - (exist_num / len(all_feat))
    important_lost_perc = -1
    if important_feat is not None:
        important_exist_num = len(important_feat.intersection(feat_set))
        important_lost_perc = 1 - important_exist_num / len(important_feat)
    return lost_perc, important_lost_perc


def lost_feature_summary_map(val, intervals):

    """
    map missing percentage to counters
    """

    feat_perc, important_feat_perc = val
    rs1, rs2 = Counter(), Counter()
    for s, e in zip(intervals[:-1], intervals[1:]):
        if s <= feat_perc < e:
            rs1[(s, e)] = 1
        if s <= important_feat_perc < e:
            rs2[(s, e)] = 1

    if feat_perc == intervals[-1]:
        rs1["1"] = 1
    if important_feat_perc == intervals[-1]:
        rs2["1"] = 1
    return rs1, rs2


def map_partitions_count(iterable, tag_id_mapping, dense_input=True, missing_val=None):

    count_arr = np.zeros(len(tag_id_mapping))
    for k, v in iterable:

        # in dense input, missing feature is set as np.nan
        if dense_input:
            feature = v.features  # a numpy array
            arr = np.array(list(feature))
            if missing_val is None:
                idx_arr = np.argwhere(~np.isnan(arr)).flatten()
            else:
                idx_arr = np.argwhere(~(arr == missing_val)).flatten()

        # in sparse input, missing features have no key in the dict
        else:
            feature = v.features.sparse_vec  # a dict
            idx_arr = np.array(list(feature.keys()))

        if len(idx_arr) != 0:
            count_arr[idx_arr] += 1

    return count_arr


def reduce_count_rs(arr1, arr2):
    return arr1 + arr2


def count_feature_ratio(tb, tag_id_mapping, dense_input, missing_val=None):
    func = functools.partial(map_partitions_count, tag_id_mapping=tag_id_mapping, dense_input=dense_input,
                             missing_val=missing_val)
    rs = tb.mapPartitions(func)
    return rs.reduce(reduce_count_rs)


def lost_feature_summary_reduce(val1, val2):

    """
    reduce counters
    """
    return val1[0] + val2[0], val1[1] + val2[1]


def lost_feature_summary(tb, interval=0.1):

    intervals = []
    assert type(interval) == float and 1 > interval > 0
    for i in range(0, 100, int(interval*100)):
        intervals.append(i/100)
    intervals.append(1.0)
    func = functools.partial(lost_feature_summary_map, intervals=intervals)
    map_table = tb.mapValues(func)
    rs = map_table.reduce(lost_feature_summary_reduce)

    # check
    for counter in rs:
        if len(counter) != 0:
            sum_ = 0
            for k in counter:
                sum_ += counter[k]
            assert sum_ == tb.count()

    return rs


class FeatureStatistic(ModelBase):

    def __init__(self, missing_val=None):
        super(FeatureStatistic, self).__init__()

        self.missing_val = None
        self.feature_summary = {}
        self.missing_feature = []
        self.all_feature_list = []
        self.tag_id_mapping, self.id_tag_mapping = {}, {}
        self.dense_missing_val = missing_val

    @staticmethod
    def is_sparse(tb):
        return type(tb.take(1)[0][1].features) == SparseVector

    @staticmethod
    def check_table_content(tb):

        if not tb.count() > 0:
            raise ValueError('input table must contains at least 1 sample')
        first_ = tb.take(1)[0][1]
        if type(first_) == Instance:
            return True
        else:
            raise ValueError('unknown input format')

    def fit(self, tb):

        LOGGER.debug('start to compute feature lost ratio')

        if not self.check_table_content(tb):
            raise ValueError('contents of input table must be instances of class “Instance"')

        header = tb.schema['header']
        self.all_feature_list = header

        self.tag_id_mapping = {v: k for k, v in enumerate(header)}
        self.id_tag_mapping = {k: v for k, v in enumerate(header)}

        feature_count_rs = count_feature_ratio(tb, self.tag_id_mapping, not self.is_sparse(tb),
                                               missing_val=self.missing_val)
        for idx, count_val in enumerate(feature_count_rs):
            self.feature_summary[self.id_tag_mapping[idx]] = 1 - (count_val/tb.count())
            if (count_val/tb.count()) == 0:
                self.missing_feature.append(self.id_tag_mapping[idx])

        return self.feature_summary
