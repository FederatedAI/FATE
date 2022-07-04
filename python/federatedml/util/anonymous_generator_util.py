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
import copy
import numpy as np
from federatedml.util.data_format_preprocess import DataFormatPreProcess


ANONYMOUS_COLUMN_PREFIX = "x"
ANONYMOUS_LABEL = "y"


class Anonymous(object):
    def __init__(self, role=None, party_id=None):
        self._role = role
        self._party_id = party_id

    @staticmethod
    def anonymous_migrate(anonymous_header, role, original_party_id, new_party_id):
        """
        split, replace party_id, then concat
        """
        pass

    def extend_columns(self, original_anonymous_header, extend_header):
        """
        一些比较复杂的情况：exp_i_0, exp_i_1... exp_k
        """
        extend_anonymous_header = []
        exp_start_idx = 0
        for anonymous_col_name in original_anonymous_header:
            if not self.is_expand_column(anonymous_col_name):
                continue

            exp_start_idx = max(exp_start_idx, self.get_expand_idx(anonymous_col_name) + 1)

        for i in range(len(extend_header)):
            extend_anonymous_header.append(self.generate_expand_anonymous_column(exp_start_idx + i))

        return original_anonymous_header + extend_anonymous_header

    @staticmethod
    def get_anonymous_header(schema):
        return schema["anonymous_header"]

    @staticmethod
    def filter_anonymous_header(schema, filter_ins):
        return schema["anonymous_header"][np.array(filter_ins)]

    @staticmethod
    def reset_anonymous_header(schema, anonymous_header):
        new_schema = copy.deepcopy(schema)
        new_schema["anonymous_header"] = anonymous_header
        return new_schema

    @staticmethod
    def generate_derived_header(original_header, original_anonymous_header, derived_dict):
        new_anonymous_header = []
        for column, anonymous_column in zip(original_header, original_anonymous_header):
            if column not in derived_dict:
                new_anonymous_header.append(anonymous_column)
            else:
                for i in range(len(derived_dict[column])):
                    new_anonymous_column = "_".join([anonymous_column, str(i)])
                    new_anonymous_header.append(new_anonymous_column)

        return new_anonymous_header

    def generate_expand_anonymous_column(self, fid):
        return "_".join(map(str, [self._role, self._party_id, "exp", fid]))

    @staticmethod
    def is_expand_column(column_name):
        splits = column_name.split("_", -1)
        return splits[-2] == "exp"

    @staticmethod
    def get_expand_idx(col_name):
        return int(col_name.split("_", -1)[-1])

    @staticmethod
    def update_anonymous_header_with_role(schema, role, party_id):
        party_id = str(party_id)
        new_schema = copy.deepcopy(schema)
        if "anonymous_header" in schema:
            old_anonymous_header = schema["anonymous_header"]
            new_anonymous_header = ["_".join([role, party_id, col_name]) for col_name in old_anonymous_header]
            new_schema["anonymous_header"] = new_anonymous_header

        if "label_name" in schema:
            new_schema["anonymous_label"] = ANONYMOUS_LABEL

    def generate_anonymous_header(self, data, schema):
        if "meta" not in schema:
            raise ValueError("Can not find data's meta, fail to generate_anonymous_header")

        new_schema = copy.deepcopy(schema)
        header = schema["header"]
        if self._role:
            anonymous_header = ["_".join(map(str, [self._role, self._party_id, ANONYMOUS_COLUMN_PREFIX + str(i)]))
                                for i in range(len(header))
                                ]
        else:
            anonymous_header = [ANONYMOUS_COLUMN_PREFIX + str(i) for i in range(len(header))]

        new_schema["anonymous_header"] = anonymous_header

        if "label_name" in schema:
            new_schema["anonymous_label"] = ANONYMOUS_LABEL

        return new_schema
