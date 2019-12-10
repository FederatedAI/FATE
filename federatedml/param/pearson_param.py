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

    def __init__(self, column_names=None, column_indexes=None):
        super().__init__()
        self.column_names = column_names
        self.column_indexes = column_indexes
        if column_names is None:
            self.column_names = []
        if column_indexes is None:
            self.column_indexes = []

    def check(self):
        if not isinstance(self.column_names, list):
            raise ValueError(f"type mismatch, column_names with type {type(self.column_names)}")
        for name in self.column_names:
            if not isinstance(name, str):
                raise ValueError(f"type mismatch, column_names with element {name}(type is {type(name)})")

        if isinstance(self.column_indexes, list):
            for idx in self.column_indexes:
                if not isinstance(idx, int):
                    raise ValueError(f"type mismatch, column_indexes with element {idx}(type is {type(idx)})")

        if isinstance(self.column_indexes, int) and self.column_indexes != -1:
            raise ValueError(f"column_indexes with type int and value {self.column_indexes}(only -1 allowed)")

        if isinstance(self.column_indexes, list) and isinstance(self.column_names, list):
            if len(self.column_indexes) == 0 and len(self.column_names) == 0:
                raise ValueError(f"provide at least one column")
