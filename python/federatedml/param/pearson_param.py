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

from federatedml.param.base_param import BaseParam


class PearsonParam(BaseParam):
    """
    param for pearson correlation

    Parameters
    ----------

    column_names : list of string
        list of column names

    column_index : list of int
        list of column index

    cross_parties : bool, default: True
        if True, calculate correlation of columns from both party

    need_run : bool
        set False to skip this party

    use_mix_rand : bool, defalut: False
        mix system random and pseudo random for quicker calculation

    calc_loca_vif : bool, default True
        calculate VIF for columns in local
    """

    def __init__(
        self,
        column_names=None,
        column_indexes=None,
        cross_parties=True,
        need_run=True,
        use_mix_rand=False,
        calc_local_vif=True,
    ):
        super().__init__()
        self.column_names = column_names
        self.column_indexes = column_indexes
        self.cross_parties = cross_parties
        self.need_run = need_run
        self.use_mix_rand = use_mix_rand
        if column_names is None:
            self.column_names = []
        if column_indexes is None:
            self.column_indexes = []
        self.calc_local_vif = calc_local_vif

    def check(self):
        if not isinstance(self.use_mix_rand, bool):
            raise ValueError(
                f"use_mix_rand accept bool type only, {type(self.use_mix_rand)} got"
            )
        if self.cross_parties and (not self.need_run):
            raise ValueError(
                f"need_run should be True(which is default) when cross_parties is True."
            )
        if not isinstance(self.column_names, list):
            raise ValueError(
                f"type mismatch, column_names with type {type(self.column_names)}"
            )
        for name in self.column_names:
            if not isinstance(name, str):
                raise ValueError(
                    f"type mismatch, column_names with element {name}(type is {type(name)})"
                )

        if isinstance(self.column_indexes, list):
            for idx in self.column_indexes:
                if not isinstance(idx, int):
                    raise ValueError(
                        f"type mismatch, column_indexes with element {idx}(type is {type(idx)})"
                    )

        if isinstance(self.column_indexes, int) and self.column_indexes != -1:
            raise ValueError(
                f"column_indexes with type int and value {self.column_indexes}(only -1 allowed)"
            )

        if self.need_run:
            if isinstance(self.column_indexes, list) and isinstance(
                self.column_names, list
            ):
                if len(self.column_indexes) == 0 and len(self.column_names) == 0:
                    raise ValueError(f"provide at least one column")
