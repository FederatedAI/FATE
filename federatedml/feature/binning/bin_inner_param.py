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

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class BinInnerParam(object):
    """
    Use to store columns related params for binning process
    """

    def __init__(self):
        self.bin_indexes = []
        self.bin_names = []
        self.col_name_maps = {}
        self.header = []
        self.transform_bin_indexes = []
        self.transform_bin_names = []
        self.category_indexes = []
        self.category_names = []

    def set_header(self, header):
        self.header = header
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

    def set_bin_all(self):
        """
        Called when user set to bin all columns
        """
        self.bin_indexes = [i for i in range(len(self.header))]
        self.bin_names = self.header

    def set_transform_all(self):
        self.transform_bin_indexes = self.bin_indexes
        self.transform_bin_names = self.bin_names
        self.transform_bin_indexes.extend(self.category_indexes)
        self.transform_bin_names.extend(self.category_names)

    def add_bin_indexes(self, bin_indexes):
        for idx in bin_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.bin_indexes:
                self.bin_indexes.append(idx)
                self.bin_names.append(self.header[idx])

    def add_bin_names(self, bin_names):
        for bin_name in bin_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.bin_indexes:
                self.bin_indexes.append(idx)
                self.bin_names.append(self.header[idx])

    def add_transform_bin_indexes(self, transform_indexes):
        for idx in transform_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.transform_bin_indexes:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_transform_bin_names(self, bin_names):
        for bin_name in bin_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.transform_bin_indexes:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_category_indexes(self, category_indexes):
        if category_indexes == -1:
            category_indexes = [i for i in range(len(self.header))]
        for idx in category_indexes:
            if idx >= len(self.header):
                LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.category_indexes:
                self.category_indexes.append(idx)
                self.category_names.append(self.header[idx])

    def add_category_names(self, category_names):
        for bin_name in category_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.category_indexes:
                self.category_indexes.append(idx)
                self.category_names.append(self.header[idx])

    @property
    def bin_cols_map(self):
        assert len(self.bin_indexes) == len(self.bin_names)
        return dict(zip(self.bin_names, self.bin_indexes))

    def encode_col_name_dict(self, col_name_dict: dict):
        encoded_result = {}
        for col_name, dict_value in col_name_dict.items():
            encoded_result[self.__encode_col_name(col_name)] = dict_value
        return encoded_result

    def __encode_col_name(self, col_name):
        col_index = self.col_name_maps.get(col_name)
        if col_index is None:
            LOGGER.warning("Encoding a non-exist column name")
            return None
        return '.'.join(['host', str(col_index)])
