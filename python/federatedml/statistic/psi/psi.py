import functools
import copy
import numpy as np
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util import consts
from federatedml.feature.fate_element_type import NoneType
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase
from federatedml.param.psi_param import PSIParam
from federatedml.util import LOGGER
from federatedml.protobuf.generated.psi_model_param_pb2 import PsiSummary, FeaturePsi
from federatedml.protobuf.generated.psi_model_meta_pb2 import PSIMeta
from federatedml.util import abnormal_detection

ROUND_NUM = 6


def map_partition_handle(iterable, feat_num=10, max_bin_num=20, is_sparse=False, missing_val=NoneType()):

    count_bin = np.zeros((feat_num, max_bin_num))
    row_idx = np.array([i for i in range(feat_num)])

    for k, v in iterable:
        # last bin is for missing value
        if is_sparse:
            feature_dict = v.features.sparse_vec
            arr = np.zeros(feat_num, dtype=np.int64) + max_bin_num - 1  # max_bin_num - 1 is the missing bin val
            arr[list(feature_dict.keys())] = list(feature_dict.values())
        else:
            arr = v.features
            arr[arr == missing_val] = max_bin_num - 1

        count_bin[row_idx, arr.astype(np.int64)] += 1

    return count_bin


def map_partition_reduce(arr1, arr2):
    return arr1 + arr2


def psi_computer(expect_counter_list, actual_counter_list, expect_sample_count, actual_sample_count):

    psi_rs = []
    for exp_counter, acu_counter in zip(expect_counter_list, actual_counter_list):
        feat_psi = {}
        for key in exp_counter:
            feat_psi[key] = psi_val(exp_counter[key] / expect_sample_count, acu_counter[key] / actual_sample_count)

        total_psi = 0
        for k in feat_psi:
            total_psi += feat_psi[k]
        feat_psi['total_psi'] = total_psi
        psi_rs.append(feat_psi)

    return psi_rs


def psi_val(expected_perc, actual_perc):

    if expected_perc == 0:
        expected_perc = 1e-6
    if actual_perc == 0:
        actual_perc = 1e-6

    return (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)


def psi_val_arr(expected_arr, actual_arr, sample_num):

    expected_arr = expected_arr / sample_num
    actual_arr = actual_arr / sample_num
    expected_arr[expected_arr == 0] = 1e-6
    actual_arr[actual_arr == 0] = 1e-6
    psi_rs = (actual_arr - expected_arr) * np.log(actual_arr / expected_arr)
    return psi_rs


def count_rs_to_dict(arrs):
    dicts = []
    for i in arrs:
        rs_dict = {}
        for k, v in enumerate(i):
            rs_dict[k] = v
        dicts.append(rs_dict)
    return dicts


def np_nan_to_nonetype(inst):

    arr = inst.features
    index = np.isnan(arr)
    if index.any():
        inst = copy.deepcopy(inst)
        arr = arr.astype(object)
        arr[index] = NoneType()
        inst.features = arr
    return inst


class PSI(ModelBase):

    def __init__(self):
        super(PSI, self).__init__()
        self.model_param = PSIParam()
        self.max_bin_num = 20
        self.tag_id_mapping = {}
        self.id_tag_mapping = {}
        self.count1, self.count2 = None, None
        self.actual_table, self.expect_table = None, None
        self.data_bin1, self.data_bin2 = None, None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.psi_rs = None
        self.total_scores = None
        self.all_feature_list = None
        self.dense_missing_val = NoneType()
        self.binning_error = consts.DEFAULT_RELATIVE_ERROR

        self.interval_perc1 = None
        self.interval_perc2 = None
        self.str_intervals = None

        self.binning_obj = None

    def _init_model(self, model: PSIParam):
        self.max_bin_num = model.max_bin_num
        self.need_run = model.need_run
        self.dense_missing_val = NoneType() if model.dense_missing_val is None else model.dense_missing_val
        self.binning_error = model.binning_error

    @staticmethod
    def check_table_content(tb):

        if not tb.count() > 0:
            raise ValueError('input table must contains at least 1 sample')
        first_ = tb.take(1)[0][1]
        if isinstance(first_, Instance):
            return True
        else:
            raise ValueError('unknown input format')

    @staticmethod
    def is_sparse(tb):
        return isinstance(tb.take(1)[0][1].features, SparseVector)

    @staticmethod
    def check_duplicates(l_):
        s = set(l_)
        recorded = set()
        new_l = []
        for i in l_:
            if i in s and i not in recorded:
                new_l.append(i)
                recorded.add(i)
        return new_l

    @staticmethod
    def get_string_interval(data_split_points, id_tag_mapping, missing_bin_idx):

        # generate string interval from bin_split_points

        feature_interval = []
        for feat_idx, interval in enumerate(data_split_points):
            idx2intervals = {}
            l_ = list(interval)
            l_[-1] = 'inf'
            l_.insert(0, '-inf')
            idx = 0
            for s, e in zip(l_[:-1], l_[1:]):

                interval_str = str(id_tag_mapping[feat_idx])

                if s != '-inf':
                    interval_str = str(np.round(s, ROUND_NUM)) + "<" + interval_str
                if e != 'inf':
                    interval_str = interval_str + "<=" + str(np.round(e, ROUND_NUM))

                idx2intervals[idx] = interval_str
                idx += 1

            idx2intervals[missing_bin_idx] = 'missing'
            feature_interval.append(idx2intervals)

        return feature_interval

    @staticmethod
    def post_process_result(rs_dict, interval_dict,):

        # convert bin idx to str intervals
        # then divide count by sample num to get percentage
        #
        rs_val_list, interval_list = [], []

        for key in sorted(interval_dict.keys()):
            corresponding_str_interval = interval_dict[key]
            val = rs_dict[key]
            rs_val_list.append(np.round(val, ROUND_NUM))
            interval_list.append(corresponding_str_interval)

        return rs_val_list, interval_list

    @staticmethod
    def count_dict_to_percentage(count_rs, sample_num):

        for c in count_rs:
            for k in c:
                c[k] = c[k] / sample_num
        return count_rs

    @staticmethod
    def convert_missing_val(table):
        new_table = table.mapValues(np_nan_to_nonetype)
        new_table.schema = table.schema
        return new_table

    def fit(self, expect_table, actual_table):

        LOGGER.info('start psi computing')

        header1 = expect_table.schema['header']
        header2 = actual_table.schema['header']

        if not set(header1) == set(header2):
            raise ValueError('table header must be the same while computing psi values')

        # baseline table should not contain empty columns
        abnormal_detection.empty_column_detection(expect_table)

        self.all_feature_list = header1

        # make sure no duplicate features
        self.all_feature_list = self.check_duplicates(self.all_feature_list)

        # kv bi-directional mapping
        self.tag_id_mapping = {v: k for k, v in enumerate(self.all_feature_list)}
        self.id_tag_mapping = {k: v for k, v in enumerate(self.all_feature_list)}

        if not self.is_sparse(expect_table):  # convert missing value: nan to NoneType
            expect_table = self.convert_missing_val(expect_table)

        if not self.is_sparse(actual_table):  # convert missing value: nan to NoneType
            actual_table = self.convert_missing_val(actual_table)

        if not(self.check_table_content(expect_table) and self.check_table_content(actual_table)):
            raise ValueError('contents of input table must be instances of class "Instance"')

        param = FeatureBinningParam(method=consts.QUANTILE, bin_num=self.max_bin_num, local_only=True,
                                    error=self.binning_error)
        binning_obj = QuantileBinning(params=param, abnormal_list=[NoneType()], allow_duplicate=False)
        binning_obj.fit_split_points(expect_table)

        data_bin, bin_split_points, bin_sparse_points = binning_obj.convert_feature_to_bin(expect_table)
        LOGGER.debug('bin split points is {}, shape is {}'.format(bin_split_points, bin_split_points.shape))
        self.binning_obj = binning_obj

        self.data_bin1 = data_bin
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points
        LOGGER.debug('expect table binning done')

        count_func1 = functools.partial(map_partition_handle,
                                        feat_num=len(self.all_feature_list),
                                        max_bin_num=self.max_bin_num + 1,  # an additional bin for missing value
                                        missing_val=self.dense_missing_val,
                                        is_sparse=self.is_sparse(self.data_bin1))

        map_rs1 = self.data_bin1.applyPartitions(count_func1)
        count1 = count_rs_to_dict(map_rs1.reduce(map_partition_reduce))

        data_bin2, bin_split_points2, bin_sparse_points2 = binning_obj.convert_feature_to_bin(actual_table)
        self.data_bin2 = data_bin2
        LOGGER.debug('actual table binning done')

        count_func2 = functools.partial(map_partition_handle,
                                        feat_num=len(self.all_feature_list),
                                        max_bin_num=self.max_bin_num + 1,  # an additional bin for missing value
                                        missing_val=self.dense_missing_val,
                                        is_sparse=self.is_sparse(self.data_bin2))

        map_rs2 = self.data_bin2.applyPartitions(count_func2)
        count2 = count_rs_to_dict(map_rs2.reduce(map_partition_reduce))

        self.count1, self.count2 = count1, count2

        LOGGER.info('psi counting done')

        # compute psi from counting result
        psi_result = psi_computer(count1, count2, expect_table.count(), actual_table.count())
        self.psi_rs = psi_result

        # get total psi score of features
        total_scores = {}
        for idx, rs in enumerate(self.psi_rs):
            feat_name = self.id_tag_mapping[idx]
            total_scores[feat_name] = rs['total_psi']
        self.total_scores = total_scores

        # id-feature mapping convert, str interval computation
        self.str_intervals = self.get_string_interval(bin_split_points, self.id_tag_mapping,
                                                      missing_bin_idx=self.max_bin_num)

        self.interval_perc1 = self.count_dict_to_percentage(copy.deepcopy(count1), expect_table.count())
        self.interval_perc2 = self.count_dict_to_percentage(copy.deepcopy(count2), actual_table.count())

        self.set_summary(self.generate_summary())
        LOGGER.info('psi computation done')

    def generate_summary(self):
        return {'psi_scores': self.total_scores}

    def export_model(self):

        if not self.need_run:
            return None

        psi_summary = PsiSummary()
        psi_summary.total_score.update(self.total_scores)

        LOGGER.debug('psi total score is {}'.format(dict(psi_summary.total_score)))

        psi_summary.model_name = consts.PSI

        feat_psi_list = []

        for id_ in self.id_tag_mapping:

            feat_psi_summary = FeaturePsi()

            feat_name = self.id_tag_mapping[id_]

            feat_psi_summary.feature_name = feat_name
            interval_psi, str_intervals = self.post_process_result(self.psi_rs[id_], self.str_intervals[id_])
            interval_perc1, _ = self.post_process_result(self.interval_perc1[id_], self.str_intervals[id_])
            interval_perc2, _ = self.post_process_result(self.interval_perc2[id_], self.str_intervals[id_])

            feat_psi_summary.psi.extend(interval_psi)
            feat_psi_summary.expect_perc.extend(interval_perc1)
            feat_psi_summary.actual_perc.extend(interval_perc2)
            feat_psi_summary.interval.extend(str_intervals)

            feat_psi_list.append(feat_psi_summary)

        psi_summary.feature_psi.extend(feat_psi_list)

        LOGGER.debug('export model done')

        meta = PSIMeta()
        meta.max_bin_num = self.max_bin_num

        return {'PSIParam': psi_summary, 'PSIMeta': meta}
