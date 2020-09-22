#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:/ /www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
################################################################################
#
#
################################################################################

# =============================================================================
# FeatureHistogram
# =============================================================================
import functools
from arch.api.utils import log_utils
from arch.api import session
from federatedml.secureprotol.iterative_affine import DeterministicIterativeAffineCiphertext
import numpy as np
import scipy.sparse as sp
import uuid

LOGGER = log_utils.getLogger()


class FastFeatureHistogram(object):
    def __init__(self):
        pass

    @staticmethod
    def calculate_histogram(data_bin, grad_and_hess, bin_split_points, cipher_split_num,
                            bin_num, node_map, valid_features, use_missing, zero_as_missing):
        LOGGER.info("bin_shape is {}, node num is {}".format(bin_split_points.shape, len(node_map)))

        # Detect length of cipher
        g, h = grad_and_hess.first()[1]
        ghlist = grad_and_hess.take(100)
        cipher_length = len(str(g.cipher))
        phrase_num = int(np.ceil(float(cipher_length) / cipher_split_num))+1
        n_final = g.n_final

        # Map-Reduce Functions
        batch_histogram_cal = functools.partial(
            FastFeatureHistogram.batch_calculate_histogram,
            node_map=node_map, bin_num=bin_num,
            phrase_num=phrase_num, cipher_split_num=cipher_split_num,
            valid_features=valid_features, use_missing=use_missing, zero_as_missing=zero_as_missing
        )
        agg_histogram = functools.partial(FastFeatureHistogram.aggregate_histogram)

        # Map-Reduce Execution
        batch_histogram_intermediate_rs = data_bin.join(grad_and_hess, lambda data_inst, g_h: (data_inst, g_h))

        # old ver code
        # batch_histogram = batch_histogram_intermediate_rs.mapPartitions2(batch_histogram_cal)
        # node_histograms = batch_histogram.reduce(agg_histogram, key_func=lambda key: key[1])

        histogram_table = batch_histogram_intermediate_rs.mapReducePartitions(batch_histogram_cal, agg_histogram)

        # aggregate matrix phase

        # old ver code
        # multiplier_vector = np.array([10**(cipher_split_num*i) for i in range(phrase_num)])  # 1 X p
        # for nid in node_histograms:
        #     b, f, p, t = node_histograms[nid][2]
        #     bin_sum_matrix4d = node_histograms[nid][0].toarray().reshape((b, f, p, t))
        #     bin_cnt_matrix = node_histograms[nid][1].toarray()
        #
        #     # b X f X p X t -> b X f X t X p : multiply along the p-axis
        #     bin_sum_matrix4d_mul = bin_sum_matrix4d.transpose((0, 1, 3, 2))*multiplier_vector
        #     # b X f X t x p -> b x f x t
        #     bin_sum_matrix3d = bin_sum_matrix4d_mul.sum(axis=3)
        #
        #     left_node_sum_matrix3d = np.cumsum(bin_sum_matrix3d, axis=0)  # accumulate : b X f X t
        #     left_node_cnt_matrix = np.cumsum(bin_cnt_matrix, axis=0)  # accumulate : b X f
        #
        #     node_histograms[nid] = [left_node_sum_matrix3d, left_node_cnt_matrix]

        map_value_func = functools.partial(FastFeatureHistogram.aggregate_matrix_phase,
                                           cipher_split_num=cipher_split_num,
                                           phrase_num=phrase_num)
        histogram_table = histogram_table.mapValues(map_value_func)

        transform_func = functools.partial(FastFeatureHistogram.table_format_transform,
                                           bin_split_points=bin_split_points,
                                           valid_features=valid_features,
                                           use_missing=use_missing,
                                           n_final=n_final)

        histogram_table = histogram_table.mapPartitions(transform_func, use_previous_behavior=False)
        return histogram_table

        # return FastFeatureHistogram.construct_table(node_histograms,
        #                                             bin_split_points=bin_split_points,
        #                                             valid_features=valid_features,
        #                                             partition=data_bin.partitions,
        #                                             use_missing=use_missing)

    @staticmethod
    def aggregate_matrix_phase(value, cipher_split_num, phrase_num):

        # aggregating encrypted text, this is a mapValues function
        b, f, p, t = value[2]
        multiplier_vector = np.array([10 ** (cipher_split_num * i) for i in range(phrase_num)])
        bin_sum_matrix4d = value[0].toarray().reshape((b, f, p, t))
        bin_cnt_matrix = value[1].toarray()

        # b X f X p X t -> b X f X t X p : multiply along the p-axis
        bin_sum_matrix4d_mul = bin_sum_matrix4d.transpose((0, 1, 3, 2)) * multiplier_vector
        # b X f X t x p -> b x f x t
        bin_sum_matrix3d = bin_sum_matrix4d_mul.sum(axis=3)

        left_node_sum_matrix3d = np.cumsum(bin_sum_matrix3d, axis=0)  # accumulate : b X f X t
        left_node_cnt_matrix = np.cumsum(bin_cnt_matrix, axis=0)  # accumulate : b X f

        return [left_node_sum_matrix3d, left_node_cnt_matrix]

    @staticmethod
    def calculate_histogram_matrix(cipher_matrix, feature_matrix, bin_num, use_missing):

        # Calculate sum of para in left node for each split points
        # Return a matrix of Bins X Feature X Phrase X type
        # C(Case) F(Feature) B(Bin) P(Phrase) T(Type: grad or hess)
        # input: cipher_matrix = t X p X c  feature_matrix = c X f

        # dimension parameter
        b = bin_num + int(use_missing)
        c = feature_matrix.shape[0]
        f = feature_matrix.shape[1]
        p = cipher_matrix.shape[1]
        t = cipher_matrix.shape[0]

        # calculation
        # Cnt Matrix
        if use_missing:
            bin_num_vector = [i for i in range(bin_num)] + [-1]  # 1 x b
        else:
            bin_num_vector = [i for i in range(bin_num)]
        bin_marker_matrix3d = np.equal.outer(bin_num_vector, feature_matrix)  # b X c X f
        bin_cnt_matrix = bin_marker_matrix3d.sum(axis=1)  # b x f
        bin_cnt_matrix_sparse = sp.csc_matrix(bin_cnt_matrix)

        # Feature Matrix
        # b*f X c
        bin_marker_matrix3d_reshape = bin_marker_matrix3d.transpose((0, 2, 1)).reshape((b * f, c))
        bin_marker_matrix3d_reshape_sparse = sp.csc_matrix(bin_marker_matrix3d_reshape)

        # Cipher Matrix
        cipher_matrix_reshape = cipher_matrix.transpose((2, 1, 0)).reshape((c, p*t))  # c x p*t

        # Feature dot Cipher
        bin_sum_matrix4d_reshape = bin_marker_matrix3d_reshape_sparse.dot(cipher_matrix_reshape)  # b*f X p*t
        bin_sum_matrix4d_sparse = sp.csc_matrix(bin_sum_matrix4d_reshape)

        dim = (b, f, p, t)
        return bin_sum_matrix4d_sparse, bin_cnt_matrix_sparse, dim

    @staticmethod
    def break_down_cipher(cipher_obj, cipher_split_num, pid):
        c_str = str(cipher_obj.cipher)
        c_len = len(c_str)
        if cipher_split_num * (pid + 1) <= c_len:
            start = c_len - cipher_split_num * (pid + 1)
            end = c_len - cipher_split_num * pid
            cipher_phase = int(c_str[start:end])
        elif cipher_split_num * pid < c_len:
            start = 0
            end = c_len - cipher_split_num * pid
            cipher_phase = int(c_str[start:end])
        else:
            cipher_phase = 0
        return cipher_phase

    @staticmethod
    def batch_calculate_histogram(kv_iterator, node_map, bin_num, phrase_num, cipher_split_num,
                                  valid_features, use_missing, zero_as_missing):

        # initialize
        data_bins_dict = {}
        grad_phrase_dict = {}
        hess_phrase_dict = {}

        # read in data
        data_record = 0
        for _, value in kv_iterator:
            data_bin, nodeid_state = value[0]
            unleaf_state, nodeid = nodeid_state
            if unleaf_state == 0 or nodeid not in node_map:
                continue
            g, h = value[1]
            nid = node_map.get(nodeid)
            if nid not in data_bins_dict:
                data_bins_dict[nid] = []

            # as most sparse point is bin-0
            # when mark it as a missing value (-1), offset it to make it sparse, to restore it to -1 here
            if not use_missing or (use_missing and not zero_as_missing):
                offset = 0
            else:
                offset = -1
            data_bins_dict[nid].append(data_bin.features.toarray()[0][valid_features] + offset)

            # Break down the cipher
            for pid in range(phrase_num):
                grad_cipher_phase = FastFeatureHistogram.break_down_cipher(g, cipher_split_num, pid)
                hess_cipher_phase = FastFeatureHistogram.break_down_cipher(h, cipher_split_num, pid)

                if nid not in grad_phrase_dict:
                    grad_phrase_dict[nid] = [[] for pid in range(phrase_num)]
                grad_phrase_dict[nid][pid].append(grad_cipher_phase)

                if nid not in hess_phrase_dict:
                    hess_phrase_dict[nid] = [[] for pid in range(phrase_num)]
                hess_phrase_dict[nid][pid].append(hess_cipher_phase)

            data_record += 1
        LOGGER.info("begin batch calculate histogram, data count is {}".format(data_record))

        # calculate histogram matrix
        ret = []
        for nid in data_bins_dict:

            feature_matrix = np.array(data_bins_dict[nid])  # c X f
            cipher_matrix = np.array([grad_phrase_dict[nid], hess_phrase_dict[nid]])  # t X p X c

            bin_sum_matrix4d_sparse, bin_cnt_matrix_sparse, dim = \
                FastFeatureHistogram.calculate_histogram_matrix(
                    cipher_matrix=cipher_matrix,
                    feature_matrix=feature_matrix,
                    bin_num=bin_num,
                    use_missing=use_missing
                )

            ret.append((nid, [bin_sum_matrix4d_sparse, bin_cnt_matrix_sparse, dim]))

        return ret

    @staticmethod
    def aggregate_histogram(histogram1, histogram2):
        bin_sum_matrix4d_sparse = histogram1[0] + histogram2[0]
        bin_cnt_matrix_sparse = histogram1[1] + histogram2[1]
        dim = histogram1[2]

        return [bin_sum_matrix4d_sparse, bin_cnt_matrix_sparse, dim]

    @staticmethod
    def get_obj(raw, n_final):
        if raw == 0:
            result = 0
        else:
            result = DeterministicIterativeAffineCiphertext(raw, n_final)
        return result

    @staticmethod
    def table_format_transform(kv_iterator, bin_split_points, valid_features, use_missing, n_final):

        ret = []
        get_obj = functools.partial(FastFeatureHistogram.get_obj, n_final=n_final)
        for nid, value in kv_iterator:
            valid_fid = 0
            for fid in range(len(valid_features)):

                if valid_features[fid]:
                    feature_bin_num = len(bin_split_points[fid]) + int(use_missing)
                    histogram = [[] for _ in range(feature_bin_num)]
                    for bid in range(len(bin_split_points[fid])):
                        grad = value[0][bid, valid_fid, 0]
                        hess = value[0][bid, valid_fid, 1]
                        cnt = value[1][bid, valid_fid]
                        histogram[bid].append(get_obj(grad))
                        histogram[bid].append(get_obj(hess))
                        histogram[bid].append(cnt)

                    if use_missing:
                        grad = value[0][-1, valid_fid, 0]
                        hess = value[0][-1, valid_fid, 1]
                        cnt = value[1][-1, valid_fid]
                        histogram[-1].append(get_obj(grad))
                        histogram[-1].append(get_obj(hess))
                        histogram[-1].append(cnt)

                    valid_fid += 1
                    # key, value
                    ret.append(((nid, fid), (fid, histogram)))
                else:
                    # empty histogram
                    ret.append(((nid, fid), (fid, [])))

        return ret

    @staticmethod
    def construct_table(histograms_dict, bin_split_points, valid_features, partition, use_missing, n_final):

        get_obj = functools.partial(FastFeatureHistogram.get_obj, n_final=n_final)
        buf = []
        for nid in histograms_dict:
            valid_fid = 0
            for fid in range(len(valid_features)):
                if valid_features[fid]:
                    feature_bin_num = len(bin_split_points[fid]) + int(use_missing)
                    histogram = [[] for _ in range(feature_bin_num)]
                    for bid in range(len(bin_split_points[fid])):
                        grad = histograms_dict[nid][0][bid, valid_fid, 0]
                        hess = histograms_dict[nid][0][bid, valid_fid, 1]
                        cnt = histograms_dict[nid][1][bid, valid_fid]
                        histogram[bid].append(get_obj(grad))
                        histogram[bid].append(get_obj(hess))
                        histogram[bid].append(cnt)

                    if use_missing:
                        grad = histograms_dict[nid][0][-1, valid_fid, 0]
                        hess = histograms_dict[nid][0][-1, valid_fid, 1]
                        cnt = histograms_dict[nid][1][-1, valid_fid]
                        histogram[-1].append(get_obj(grad))
                        histogram[-1].append(get_obj(hess))
                        histogram[-1].append(cnt)

                    buf.append(((nid, fid), (fid, histogram)))
                    valid_fid += 1

        return session.parallelize(buf, include_key=True, partition=partition)
