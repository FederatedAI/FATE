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


class Bucket(object):
    def __init__(self, idx=-1, adjustment_factor=0.5, right_bound=-math.inf):
        self.idx = idx
        self.left_bound = math.inf
        self.right_bound = right_bound
        self.left_neighbor_idx = idx - 1
        self.right_neighbor_idx = idx + 1
        self.event_count = 0
        self.non_event_count = 0
        self.adjustment_factor = adjustment_factor
        self.event_total = None
        self.non_event_total = None

    def set_left_neighbor(self, left_idx):
        self.left_neighbor_idx = left_idx

    def set_right_neighbor(self, right_idx):
        self.right_neighbor_idx = right_idx

    @property
    def is_mixed(self):
        return self.event_count > 0 and self.non_event_count > 0

    @property
    def total_count(self):
        return self.event_count + self.non_event_count

    def merge(self, other):
        if other is None:
            return
        if other.left_bound < self.left_bound:
            self.left_bound = other.left_bound
        if other.right_bound > self.right_bound:
            self.right_bound = other.right_bound
        self.event_count += other.event_count
        self.non_event_count += other.non_event_count
        return self

    def add(self, label, value):
        if label == 1:
            self.event_count += 1
        else:
            self.non_event_count += 1

        if value < self.left_bound:
            self.left_bound = value
        if value > self.right_bound:
            self.right_bound = value

    @property
    def iv(self):
        if self.event_total is None or self.non_event_total is None:
            raise AssertionError("Bucket's event_total or non_event_total has not been assigned")
        # only have EVENT records or Non-Event records
        if self.event_count == 0 or self.non_event_count == 0:
            event_rate = 1.0 * (self.event_count + self.adjustment_factor) / self.event_total
            non_event_rate = 1.0 * (self.non_event_count + self.adjustment_factor) / self.non_event_total
        else:
            event_rate = 1.0 * self.event_count / self.event_total
            non_event_rate = 1.0 * self.non_event_count / self.non_event_total
        woe = math.log(non_event_rate / event_rate)
        return (non_event_rate - event_rate) * woe

    @property
    def gini(self):
        if self.total_count == 0:
            return 0

        return 1 - (1.0 * self.event_count / self.total_count) ** 2 - \
              (1.0 * self.non_event_count / self.total_count) ** 2








