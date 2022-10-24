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


import collections
from sortedcontainers import SortedList


class Comparision(object):
    def __init__(self, size):
        self._histograms = collections.deque(maxlen=size)
        self._sorted_hist = SortedList()

    def add(self, value):
        if len(self._histograms) == self._histograms.maxlen:
            self._sorted_hist.remove(self._histograms[0])

        self._histograms.append(value)
        self._sorted_hist.add(value)

    def _get_lt_count(self, value):
        return self._sorted_hist.bisect_left(value=value)

    def _get_le_count(self, value):
        return self._sorted_hist.bisect_right(value=value)

    def _get_size(self):
        return len(self._histograms)

    def get_rate(self, value):
        return self._get_lt_count(value) / self._get_size()

    def is_topk(self, value, k):
        if self._get_size() <= k:
            return True

        return self._get_size() - self._get_le_count(value) < k
