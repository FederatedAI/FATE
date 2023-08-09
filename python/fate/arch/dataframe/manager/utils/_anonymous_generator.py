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
import pandas as pd


ANONYMOUS_COLUMN_PREFIX = "x"
ANONYMOUS_LABEL = "y"
ANONYMOUS_WEIGHT = "weight"
SPLICES = "_"
ANONYMOUS_ROLE = "AnonymousRole"
ANONYMOUS_PARTY_ID = "AnonymousPartyId"


class AnonymousGenerator(object):
    def __init__(self, role=None, party_id=None):
        self._role = role
        self._party_id = party_id

    def _generate_anonymous_column(self, suf):
        if self._role and self._party_id:
            return SPLICES.join([self._role, self._party_id, suf])
        else:
            return SPLICES.join([ANONYMOUS_ROLE, ANONYMOUS_PARTY_ID, suf])

    def generate_anonymous_names(self, schema):
        column_len = len(schema.columns.tolist())
        anonymous_label_name = None
        anonymous_weight_name = None

        anonymous_columns = [self._generate_anonymous_column(
            ANONYMOUS_COLUMN_PREFIX + str(i)) for i in range(column_len)]

        if schema.label_name:
            anonymous_label_name = self._generate_anonymous_column(ANONYMOUS_LABEL)

        if schema.weight_name:
            anonymous_weight_name = self._generate_anonymous_column(ANONYMOUS_WEIGHT)

        return dict(
            anonymous_label_name=anonymous_label_name,
            anonymous_weight_name=anonymous_weight_name,
            anonymous_columns=anonymous_columns,
            anonymous_summary=dict(column_len=column_len,
                                   role=self._role,
                                   party_id=self._party_id)
        )

    def _check_role_party_id_consistency(self, anonymous_summary):
        anonymous_role = anonymous_summary["role"]
        anonymous_party_id = anonymous_summary["party_id"]

        if anonymous_role and self._role is not None and anonymous_role != self._role:
            raise ValueError(f"previous_role={anonymous_role} != current_role={self._role}")

        if anonymous_party_id and self._party_id is not None and anonymous_party_id != self._party_id:
            raise ValueError(f"previous_party_id={anonymous_party_id} != current_role={self._party_id}")

    def add_anonymous_label(self):
        return self._generate_anonymous_column(ANONYMOUS_LABEL)

    def add_anonymous_weight(self):
        return self._generate_anonymous_column(ANONYMOUS_WEIGHT)

    def add_anonymous_columns(self, columns, anonymous_summary: dict):
        self._check_role_party_id_consistency(anonymous_summary)
        anonymous_summary = copy.deepcopy(anonymous_summary)

        column_len = anonymous_summary["column_len"]
        anonymous_columns = [self._generate_anonymous_column(ANONYMOUS_COLUMN_PREFIX + str(i + column_len))
                             for i in range(len(columns))]

        anonymous_summary["column_len"] = column_len + len(columns)
        return anonymous_columns, anonymous_summary

    def fill_role_and_party_id(self, anonymous_label_name, anonymous_weight_name,
                               anonymous_columns, anonymous_summary):
        anonymous_summary = copy.deepcopy(anonymous_summary)

        self._check_role_party_id_consistency(anonymous_summary)

        if anonymous_summary["role"] is None and anonymous_summary["party_id"] is None:
            anonymous_label_name = self._fill_role_and_party_id(anonymous_label_name)
            anonymous_weight_name = self._fill_role_and_party_id(anonymous_weight_name)
            anonymous_columns = self._fill_role_and_party_id(anonymous_columns)
            anonymous_summary["role"] = self._role
            anonymous_summary["party_id"] = self._party_id

        return dict(
            anonymous_label_name=anonymous_label_name,
            anonymous_weight_name=anonymous_weight_name,
            anonymous_columns=anonymous_columns,
            anonymous_summary=anonymous_summary
        )

    def _fill_role_and_party_id(self, name):
        if name is None:
            return name

        if isinstance(name, str):
            role, party_id, suf = name.split(SPLICES, 2)
            if role != ANONYMOUS_ROLE or party_id != ANONYMOUS_PARTY_ID:
                raise ValueError(f"To fill anonymous names with role and party_id, it shouldn't be fill before")
            return self._generate_anonymous_column(suf)
        else:
            name = list(name)
            ret = []
            for _name in name:
                role, party_id, suf = _name.split(SPLICES, 2)
                if role != ANONYMOUS_ROLE or party_id != ANONYMOUS_PARTY_ID:
                    raise ValueError(f"To fill anonymous names with role and party_id, it shouldn't be fill before")

                ret.append(self._generate_anonymous_column(suf))

            return pd.Index(ret)
