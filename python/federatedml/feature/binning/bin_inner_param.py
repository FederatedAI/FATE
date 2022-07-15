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


import copy

from federatedml.util import LOGGER


class BinInnerParam(object):
    """
    Use to store columns related params for binning process
    """

    def __init__(self):
        self.bin_indexes = []
        self.bin_names = []
        self.bin_indexes_added_set = set()
        self.col_name_maps = {}
        self.anonymous_col_name_maps = {}
        self.col_name_anonymous_maps = {}
        self.header = []
        self.anonymous_header = []
        self.transform_bin_indexes = []
        self.transform_bin_names = []
        self.transform_bin_indexes_added_set = set()
        self.category_indexes = []
        self.category_names = []
        self.category_indexes_added_set = set()

    def set_header(self, header, anonymous_header):
        self.header = copy.deepcopy(header)
        self.anonymous_header = copy.deepcopy(anonymous_header)
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

        self.anonymous_col_name_maps = dict(zip(self.anonymous_header, self.header))
        self.col_name_anonymous_maps = dict(zip(self.header, self.anonymous_header))

    def set_bin_all(self):
        """
        Called when user set to bin all columns
        """
        self.bin_indexes = [i for i in range(len(self.header))]
        self.bin_indexes_added_set = set(self.bin_indexes)
        self.bin_names = copy.deepcopy(self.header)

    def set_transform_all(self):
        self.transform_bin_indexes = self.bin_indexes
        self.transform_bin_names = self.bin_names
        self.transform_bin_indexes.extend(self.category_indexes)
        self.transform_bin_names.extend(self.category_names)
        self.transform_bin_indexes_added_set = set(self.transform_bin_indexes)

    def add_bin_indexes(self, bin_indexes):
        if bin_indexes is None:
            return
        for idx in bin_indexes:
            if idx >= len(self.header):
                # LOGGER.warning("Adding a index that out of header's bound")
                # continue
                raise ValueError("Adding a index that out of header's bound")
            if idx not in self.bin_indexes_added_set:
                self.bin_indexes.append(idx)
                self.bin_indexes_added_set.add(idx)
                self.bin_names.append(self.header[idx])

    def add_bin_names(self, bin_names):
        if bin_names is None:
            return

        for bin_name in bin_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.bin_indexes_added_set:
                self.bin_indexes.append(idx)
                self.bin_indexes_added_set.add(idx)
                self.bin_names.append(self.header[idx])

    def add_transform_bin_indexes(self, transform_indexes):
        if transform_indexes is None:
            return

        for idx in transform_indexes:
            if idx >= len(self.header) or idx < 0:
                raise ValueError("Adding a index that out of header's bound")
                # LOGGER.warning("Adding a index that out of header's bound")
                # continue
            if idx not in self.transform_bin_indexes_added_set:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_indexes_added_set.add(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_transform_bin_names(self, transform_names):
        if transform_names is None:
            return
        for bin_name in transform_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                raise ValueError("Adding a col_name that is not exist in header")

            if idx not in self.transform_bin_indexes_added_set:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_indexes_added_set.add(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_category_indexes(self, category_indexes):
        if category_indexes == -1:
            category_indexes = [i for i in range(len(self.header))]
        elif category_indexes is None:
            return

        for idx in category_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.category_indexes_added_set:
                self.category_indexes.append(idx)
                self.category_indexes_added_set.add(idx)
                self.category_names.append(self.header[idx])

            if idx in self.bin_indexes_added_set:
                self.bin_indexes_added_set.remove(idx)

        self._align_bin_index()

    def add_category_names(self, category_names):
        if category_names is None:
            return

        for bin_name in category_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.category_indexes_added_set:
                self.category_indexes.append(idx)
                self.category_indexes_added_set.add(idx)
                self.category_names.append(self.header[idx])
            if idx in self.bin_indexes_added_set:
                self.bin_indexes_added_set.remove(idx)

        self._align_bin_index()

    def _align_bin_index(self):
        if len(self.bin_indexes_added_set) != len(self.bin_indexes):
            new_bin_indexes = []
            new_bin_names = []
            for idx in self.bin_indexes:
                if idx in self.bin_indexes_added_set:
                    new_bin_indexes.append(idx)
                    new_bin_names.append(self.header[idx])

            self.bin_indexes = new_bin_indexes
            self.bin_names = new_bin_names

    def get_need_cal_iv_cols_map(self):
        names = self.bin_names + self.category_names
        indexs = self.bin_indexes + self.category_indexes
        assert len(names) == len(indexs)
        return dict(zip(names, indexs))

    @property
    def bin_cols_map(self):
        assert len(self.bin_indexes) == len(self.bin_names)
        return dict(zip(self.bin_names, self.bin_indexes))

    @staticmethod
    def change_to_anonymous(col_name, v, col_name_anonymous_maps: dict):
        anonymous_col = col_name_anonymous_maps.get(col_name)
        return anonymous_col, v

    def get_anonymous_col_name_list(self, col_name_list: list):
        result = []
        for x in col_name_list:
            result.append(self.col_name_anonymous_maps[x])
        return result

    def get_col_name_by_anonymous(self, anonymous_col_name: str):
        return self.anonymous_col_name_maps.get(anonymous_col_name)
