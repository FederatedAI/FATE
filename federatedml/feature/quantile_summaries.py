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
#

import math

from federatedml.util import consts


class Stats(object):
    def __init__(self, value, g: int, delta: int):
        self.value = value
        self.g = g
        self.delta = delta


class QuantileSummaries(object):
    def __init__(self, compress_thres=consts.DEFAULT_COMPRESS_THRESHOLD,
                 head_size=consts.DEFAULT_HEAD_SIZE,
                 error=consts.DEFAULT_RELATIVE_ERROR):
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.head_sampled = []
        self.sampled = []  # list of Stats
        self.count = 0  # Total observations appeared

    # insert a number
    def insert(self, x):
        """
        Insert an observation of data. First store in a array buffer. If the buffer is full,
        do a batch insert. If the size of sampled list reach compress_thres, compress this list.
        Parameters
        ----------
        x : float
            The observation that prepare to insert

        """
        self.head_sampled.append(x)
        if len(self.head_sampled) >= self.head_size:
            self._insert_head_buffer()
            if len(self.sampled) >= self.compress_thres:
                self.compress()

    def _insert_head_buffer(self):
        if not len(self.head_sampled):  # If empty
            return
        current_count = self.count
        sorted_head = sorted(self.head_sampled)
        new_sampled = []
        sample_idx = 0
        ops_idx = 0
        while ops_idx < len(sorted_head):
            current_sample = sorted_head[ops_idx]
            while sample_idx < len(self.sampled) and self.sampled[sample_idx].value <= current_sample:
                new_sampled.append(self.sampled[sample_idx])
                sample_idx += 1

            current_count += 1

            # If it is the first one to insert or if it is the last one
            if not new_sampled or (sample_idx == len(self.sampled) and
                                           ops_idx == len(sorted_head) - 1):
                delta = 0
            else:
                delta = math.floor(2 * self.error * current_count) - 1

            new_stats = Stats(current_sample, 1, delta)
            new_sampled.append(new_stats)
            ops_idx += 1
        self.sampled = new_sampled
        self.head_sampled = []
        self.count = current_count

    def compress(self):
        self._insert_head_buffer()
        merge_threshold = math.floor(2 * self.error * self.count) - 1
        compressed = self._compress_immut(merge_threshold)
        self.sampled = compressed

    def merge(self, other):
        """
        merge current summeries with the other one.
        Parameters
        ----------
        other : QuantileSummaries
            The summaries to be merged
        """
        if other.head_sampled:
            other._insert_head_buffer()

        if self.head_sampled:
            self._insert_head_buffer()

        if other.count == 0:
            return self

        if self.count == 0:
            return other

        # merge two sorted array
        new_sample = []
        i, j = 0, 0
        while i < len(self.sampled) and j < len(other.sampled):
            if self.sampled[i].value < other.sampled[j].value:
                new_sample.append(self.sampled[i])
                i += 1
            else:
                new_sample.append(other.sampled[j])
                j += 1
        new_sample += self.sampled[i:]
        new_sample += other.sampled[j:]

        self.sampled = new_sample
        self.count += other.count
        merge_threshold = math.floor(2 * self.error * self.count) - 1
        self.sampled = self._compress_immut(merge_threshold)

    def query(self, quantile):
        """
        Given the queried quantile, return the approximation guaranteed result
        Parameters
        ----------
        quantile : float [0.0, 1.0]
            The target quantile

        Returns
        -------
        float, the corresponding value result.
        """
        if self.head_sampled:
            self._insert_head_buffer()

        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be in range [0.0, 1.0]")

        if quantile <= self.error:
            return self.sampled[0].value

        if quantile >= 1 - self.error:
            return self.sampled[-1].value

        rank = math.ceil(quantile * self.count)
        target_error = math.ceil(self.error * self.count)
        min_rank = 0
        i = 1
        while i < len(self.sampled) - 1:
            cur_sample = self.sampled[i]
            min_rank += cur_sample.g
            max_rank = min_rank + cur_sample.delta
            if max_rank - target_error <= rank <= min_rank + target_error:
                return cur_sample.value
            i += 1
        return self.sampled[-1].value

    def _compress_immut(self, merge_threshold):
        if not self.sampled:
            return self.sampled

        res = []

        # Start from the last element
        head = self.sampled[-1]
        i = len(self.sampled) - 2  # Do not merge the last element

        while i >= 1:
            this_sample = self.sampled[i]
            if this_sample.g + head.g + head.delta < merge_threshold:
                head.g = head.g + this_sample.g
            else:
                res.append(head)
                head = this_sample
            i -= 1
        res.append(head)

        # If head of current sample is smaller than this new res's head
        # Add current head into res
        current_head = self.sampled[0]
        if current_head.value <= head.value and len(self.sampled) > 1:
            res.append(current_head)

        # Python do not support prepend, thus, use reverse instead
        res.reverse()
        return res


