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
SPLICES = "_"


class Anonymous(object):
    def __init__(self, role=None, party_id=None, migrate_mapping=None):
        self._role = role
        self._party_id = party_id
        self._migrate_mapping = migrate_mapping

    def migrate_schema_anonymous(self, schema):
        if "anonymous_header" in schema:
            schema["anonymous_header"] = self.migrate_anonymous(schema["anonymous_header"])

        if "anonymous_label" in schema:
            schema["anonymous_label"] = self.migrate_anonymous(schema['anonymous_label'])

        return schema

    def migrate_anonymous(self, anonymous_header):
        ret_list = True
        if not isinstance(anonymous_header, list):
            ret_list = False
            anonymous_header = [anonymous_header]

        migrate_anonymous_header = []
        for column in anonymous_header:
            role, party_id, suf = column.split(SPLICES, 2)
            try:
                migrate_party_id = self._migrate_mapping[role][int(party_id)]
            except KeyError:
                migrate_party_id = self._migrate_mapping[role][party_id]
            except BaseException:
                migrate_party_id = None

            if migrate_party_id is not None:
                migrate_anonymous_header.append(self.generate_anonymous_column(role, migrate_party_id, suf))
            else:
                migrate_anonymous_header.append(column)

        if not ret_list:
            migrate_anonymous_header = migrate_anonymous_header[0]

        return migrate_anonymous_header

    def is_anonymous(self, column):
        splits = self.get_anonymous_column_splits(column)
        if len(splits) < 3:
            return False
        role, party_id = splits[0], splits[1]

        return role in self._migrate_mapping and int(party_id) in self._migrate_mapping[role]

    def extend_columns(self, original_anonymous_header, extend_header):
        extend_anonymous_header = []
        exp_start_idx = 0
        for anonymous_col_name in original_anonymous_header:
            if not self.is_expand_column(anonymous_col_name):
                continue

            exp_start_idx = max(exp_start_idx, self.get_expand_idx(anonymous_col_name) + 1)

        for i in range(len(extend_header)):
            extend_anonymous_header.append(self.__generate_expand_anonymous_column(exp_start_idx + i))

        return original_anonymous_header + extend_anonymous_header

    @staticmethod
    def get_party_id_from_anonymous_column(anonymous_column):
        splits = Anonymous.get_anonymous_column_splits(anonymous_column)
        if len(splits) < 3:
            raise ValueError("This is not a anonymous_column")
        return splits[1]

    @staticmethod
    def get_role_from_anonymous_column(anonymous_column):
        splits = Anonymous.get_anonymous_column_splits(anonymous_column)
        if len(splits) < 3:
            raise ValueError("This is not a anonymous_column")
        return splits[0]

    @staticmethod
    def get_suffix_from_anonymous_column(anonymous_column):
        splits = Anonymous.get_anonymous_column_splits(anonymous_column, num=2)
        if len(splits) < 3:
            raise ValueError("This is not a anonymous_column")
        return splits[-1]

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
                    new_anonymous_column = SPLICES.join([anonymous_column, str(i)])
                    new_anonymous_header.append(new_anonymous_column)

        return new_anonymous_header

    def __generate_expand_anonymous_column(self, fid):
        return SPLICES.join(map(str, [self._role, self._party_id, "exp", fid]))

    @staticmethod
    def generate_anonymous_column(role, party_id, suf):
        return SPLICES.join([role, str(party_id), suf])

    @staticmethod
    def get_anonymous_column_splits(column, num=-1):
        return column.split(SPLICES, num)

    @staticmethod
    def is_expand_column(column_name):
        splits = Anonymous.get_anonymous_column_splits(column_name)
        return splits[-2] == "exp"

    @staticmethod
    def get_expand_idx(column_name):
        return int(Anonymous.get_anonymous_column_splits(column_name)[-1])

    @staticmethod
    def update_anonymous_header_with_role(schema, role, party_id):
        party_id = str(party_id)
        new_schema = copy.deepcopy(schema)
        if "anonymous_header" in schema:
            old_anonymous_header = schema["anonymous_header"]
            new_anonymous_header = [Anonymous.generate_anonymous_column(role, party_id, col_name)
                                    for col_name in old_anonymous_header]
            new_schema["anonymous_header"] = new_anonymous_header

        if "label_name" in schema:
            new_schema["anonymous_label"] = Anonymous.generate_anonymous_column(role, party_id, ANONYMOUS_LABEL)

        return new_schema

    def generate_anonymous_header(self, schema):
        new_schema = copy.deepcopy(schema)
        header = schema["header"]
        if self._role:
            anonymous_header = [Anonymous.generate_anonymous_column(self._role,
                                                                    self._party_id,
                                                                    ANONYMOUS_COLUMN_PREFIX + str(i))
                                for i in range(len(header))]
        else:
            anonymous_header = [ANONYMOUS_COLUMN_PREFIX + str(i) for i in range(len(header))]

        new_schema["anonymous_header"] = anonymous_header

        if "label_name" in schema:
            if self._role:
                new_schema["anonymous_label"] = self.generate_anonymous_column(self._role,
                                                                               self._party_id,
                                                                               ANONYMOUS_LABEL)
            else:
                new_schema["anonymous_label"] = ANONYMOUS_LABEL

        return new_schema

    def generated_compatible_anonymous_header_with_old_version(self, header):
        if self._role is None or self._party_id is None:
            raise ValueError("Please init anonymous generator with role & party_id")
        return [SPLICES.join([self._role, str(self._party_id), str(idx)]) for idx in range(len(header))]

    @staticmethod
    def is_old_version_anonymous_header(anonymous_header):
        for anonymous_col in anonymous_header:
            splits = anonymous_col.split(SPLICES, -1)
            if len(splits) != 3:
                return False

            try:
                index = int(splits[2])
            except ValueError:
                return False

        return True
