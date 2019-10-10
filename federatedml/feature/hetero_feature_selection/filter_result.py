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

import operator


class FilterResult(object):
    def __init__(self):
        self._left_cols = {}
        self._pass_filter_nums = {}

    @property
    def this_to_select_cols(self):
        return [x for x, is_left in self._left_cols.items() if is_left]

    def get_left_cols(self):
        return self._left_cols

    def set_left_cols(self, left_cols):
        self._left_cols = left_cols

    def get_sorted_col_names(self):
        result = sorted(self._pass_filter_nums.items(), key=operator.itemgetter(1), reverse=True)
        return [x for x, _ in result]


class SelfFilterResult(FilterResult):
    """
    Store all the filtered results
    """

    def __init__(self, header, to_select_cols_all):
        super().__init__()
        self.__header = tuple(header)
        self.__to_select_cols_all = tuple(to_select_cols_all)
        self.__header_index = {col_name: col_index for col_index, col_name in enumerate(header)}
        self._left_cols = {x: True for x in to_select_cols_all}
        self._pass_filter_nums = {x: 0 for x in to_select_cols_all}

    @property
    def this_to_select_cols_index(self):
        return [self.__header_index[col_name] for col_name in self.this_to_select_cols]

    def add_left_cols(self, left_cols: dict):
        for col_name, is_left in left_cols.items():
            if is_left:
                self._pass_filter_nums[col_name] += 1
        self._left_cols = left_cols

    def add_left_col_index(self, left_col_index: dict):
        for col_index, is_left in left_col_index.items():
            col_name = self.__header[col_index]
            self._left_cols[col_name] = is_left
            if is_left:
                self._pass_filter_nums[col_name] += 1


class RemoteFilterResult(FilterResult):
    """
    Store remote party filter results
    Typically used in Guest party to store Host party results
    """

    def __init__(self):
        super().__init__()
        self.to_select_cols_dict = {}

    def set_to_select_cols(self, this_to_select_cols_index):
        self.to_select_cols_dict = {col_index: True for col_index in this_to_select_cols_index}

    def add_left_cols(self, left_cols):
        for col_index, is_left in left_cols.items():
            col_name = 'host.' + str(col_index)
            if col_name not in self._pass_filter_nums:
                self._pass_filter_nums[col_name] = 0
            if is_left:
                self._pass_filter_nums[col_name] += 1
        self._left_cols = left_cols
