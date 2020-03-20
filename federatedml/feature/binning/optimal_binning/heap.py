#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math

import numpy as np

from arch.api.utils import log_utils
from federatedml.feature.binning.optimal_binning.bucket_info import Bucket
from federatedml.param.feature_binning_param import OptimalBinningParam

LOGGER = log_utils.getLogger()


class HeapNode(object):
    def __init__(self):
        self.left_bucket: Bucket = None
        self.right_bucket: Bucket = None
        self.event_count = 0
        self.non_event_count = 0

        self.score = None

    def cal_score(self):
        raise NotImplementedError("Should not call here")

    @property
    def total_count(self):
        return self.event_count + self.non_event_count


class IvHeapNode(HeapNode):
    def __init__(self, adjustment_factor=0.5):
        super().__init__()

        self.adjustment_factor = adjustment_factor
        self.event_total = 0
        self.non_event_total = 0

    def cal_score(self):
        """
        IV = ∑(py_i - pn_i ) * WOE
        where py_i is event_rate, pn_i is non_event_rate
        WOE = log(non_event_rate / event_rate)
        """

        self.event_count = self.left_bucket.event_count + self.right_bucket.event_count
        self.non_event_count = self.left_bucket.non_event_count + self.right_bucket.non_event_count
        if self.total_count == 0:
            self.score = -math.inf
            return

        # if self.left_bucket.left_bound != math.inf and self.right_bucket.right_bound != -math.inf:
        #     if (self.left_bucket.left_bound <= self.right_bucket.right_bound):
        #         self.score = -math.inf
        #         return

        self.event_total = self.left_bucket.event_total
        self.non_event_total = self.left_bucket.non_event_total

        if self.event_count == 0 or self.non_event_count == 0:
            event_rate = 1.0 * (self.event_count + self.adjustment_factor) / self.event_total
            non_event_rate = 1.0 * (self.non_event_count + self.adjustment_factor) / self.non_event_total
        else:
            event_rate = 1.0 * self.event_count / self.event_total
            non_event_rate = 1.0 * self.non_event_count / self.non_event_total
        merge_woe = math.log(event_rate / non_event_rate)

        merge_iv = (event_rate - non_event_rate) * merge_woe
        self.score = self.left_bucket.iv + self.right_bucket.iv - merge_iv


class GiniHeapNode(HeapNode):
    def cal_score(self):
        """
        gini = 1 - ∑(p_i^2 ) = 1 -（event / total）^2 - (nonevent / total)^2
        """

        self.event_count = self.left_bucket.event_count + self.right_bucket.event_count
        self.non_event_count = self.left_bucket.non_event_count + self.right_bucket.non_event_count
        if self.total_count == 0:
            self.score = -math.inf
            return

        # if self.total_count == 0 or self.left_bucket.left_bound == self.right_bucket.right_bound:
        #     self.score = -math.inf
        #     return
        merged_gini = 1 - (1.0 * self.event_count / self.total_count) ** 2 - \
                      (1.0 * self.non_event_count / self.total_count) ** 2
        self.score = merged_gini - self.left_bucket.gini - self.right_bucket.gini


class ChiSquareHeapNode(HeapNode):
    def cal_score(self):
        """
        X^2 = ∑∑(A_ij - E_ij )^2 / E_ij
        where E_ij = (N_i / N) * C_j. N is total count of merged bucket, N_i is the total count of ith bucket
        and C_j is the count of jth label in merged bucket.
        A_ij is number of jth label in ith bucket.
        """

        self.event_count = self.left_bucket.event_count + self.right_bucket.event_count
        self.non_event_count = self.left_bucket.non_event_count + self.right_bucket.non_event_count
        if self.total_count == 0:
            self.score = -math.inf
            return

        c1 = self.left_bucket.event_count + self.right_bucket.event_count
        c0 = self.left_bucket.non_event_count + self.right_bucket.non_event_count

        if c1 == 0 or c0 == 0:
            self.score = - math.inf
            return

        e_left_1 = (self.left_bucket.total_count / self.total_count) * c1
        e_left_0 = (self.left_bucket.total_count / self.total_count) * c0
        e_right_1 = (self.right_bucket.total_count / self.total_count) * c1
        e_right_0 = (self.right_bucket.total_count / self.total_count) * c0

        chi_square = np.square(self.left_bucket.event_count - e_left_1) / e_left_1 + \
                     np.square(self.left_bucket.non_event_count - e_left_0) / e_left_0 + \
                     np.square(self.right_bucket.event_count - e_right_1) / e_right_1 + \
                     np.square(self.right_bucket.non_event_count - e_right_0) / e_right_0
        LOGGER.debug("chi_sqaure: {}".format(chi_square))

        self.score = chi_square


def heap_node_factory(optimal_param: OptimalBinningParam, left_bucket=None, right_bucket=None):
    metric_method = optimal_param.metric_method
    if metric_method == 'iv':
        node = IvHeapNode(adjustment_factor=optimal_param.adjustment_factor)
    elif metric_method == 'gini':
        node = GiniHeapNode()
    elif metric_method == 'chi_square':
        node = ChiSquareHeapNode()
    else:
        raise ValueError("metric_method: {} cannot recognized".format(metric_method))

    if left_bucket is not None:
        node.left_bucket = left_bucket

    if right_bucket is not None:
        node.right_bucket = right_bucket

    if (left_bucket and right_bucket) is not None:
        node.cal_score()
    else:
        LOGGER.warning("In heap factory, left_bucket is {}, right bucket is {}, not all of them has been assign".format(
            left_bucket, right_bucket
        ))

    return node


class MinHeap(object):
    def __init__(self):
        self.size = 0
        self.node_list = []

    @property
    def is_empty(self):
        return self.size <= 0

    def insert(self, heap_node: HeapNode):
        self.size += 1
        self.node_list.append(heap_node)
        self._move_up(self.size - 1)

    def pop(self):
        min_node = self.node_list[0] if not self.is_empty else None

        if min_node is not None:
            self.node_list[0] = self.node_list[self.size - 1]
            self.node_list.pop()
            self.size -= 1
            self._move_down(0)
        return min_node

    def _switch_node(self, idx_1, idx_2):
        if idx_1 >= self.size or idx_2 >= self.size:
            return
        self.node_list[idx_1], self.node_list[idx_2] = self.node_list[idx_2], self.node_list[idx_1]

    @staticmethod
    def _get_parent_index(index):
        if index == 0:
            return None
        parent_index = (index - 1) / 2
        return int(parent_index) if parent_index >= 0 else None

    def _get_left_child_idx(self, idx):
        child_index = (2 * idx) + 1
        return child_index if child_index < self.size else None

    def _get_right_child_idx(self, idx):
        child_index = (2 * idx) + 2
        return child_index if child_index < self.size else None

    def _move_down(self, curt_idx):
        if curt_idx >= self.size:
            return

        min_idx = curt_idx
        while True:
            left_child_idx = self._get_left_child_idx(curt_idx)
            right_child_idx = self._get_right_child_idx(curt_idx)

            if left_child_idx is not None and self.node_list[left_child_idx].score < self.node_list[curt_idx].score:
                min_idx = left_child_idx

            if right_child_idx is not None and self.node_list[right_child_idx].score < self.node_list[min_idx].score:
                min_idx = right_child_idx

            if min_idx != curt_idx:
                self._switch_node(curt_idx, min_idx)
                curt_idx = min_idx
            else:
                break

    def _move_up(self, curt_idx):
        if curt_idx >= self.size:
            return
        while True:
            parent_idx = self._get_parent_index(curt_idx)
            if parent_idx is None:
                break

            if self.node_list[curt_idx].score < self.node_list[parent_idx].score:
                self._switch_node(curt_idx, parent_idx)
                curt_idx = parent_idx
            else:
                break
