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
DEFAULT_SITE_NAME = "AnonymousRole_AnonymousPartyId"


class AnonymousGenerator(object):
    def __init__(self, site_name=None):
        self._site_name = site_name

    def _generate_anonymous_column(self, suf):
        if self._site_name:
            return SPLICES.join([self._site_name, suf])
        else:
            return SPLICES.join([DEFAULT_SITE_NAME, suf])

    def generate_anonymous_names(self, schema):
        column_len = len(schema.columns.tolist())
        anonymous_label_name = None
        anonymous_weight_name = None

        anonymous_columns = [
            self._generate_anonymous_column(ANONYMOUS_COLUMN_PREFIX + str(i)) for i in range(column_len)
        ]

        if schema.label_name:
            anonymous_label_name = self._generate_anonymous_column(ANONYMOUS_LABEL)

        if schema.weight_name:
            anonymous_weight_name = self._generate_anonymous_column(ANONYMOUS_WEIGHT)

        return dict(
            anonymous_label_name=anonymous_label_name,
            anonymous_weight_name=anonymous_weight_name,
            anonymous_columns=anonymous_columns,
            anonymous_summary=dict(column_len=column_len, site_name=self._site_name),
        )

    def _check_site_name_consistency(self, anonymous_summary):
        anonymous_site_name = anonymous_summary["site_name"]

        if anonymous_site_name and self._site_name is not None and anonymous_site_name != self._site_name:
            raise ValueError(f"previous_site_name={anonymous_site_name} != current_site_name={self._site_name}")

    def add_anonymous_label(self):
        return self._generate_anonymous_column(ANONYMOUS_LABEL)

    def add_anonymous_weight(self):
        return self._generate_anonymous_column(ANONYMOUS_WEIGHT)

    def add_anonymous_columns(self, columns, anonymous_summary: dict):
        self._check_site_name_consistency(anonymous_summary)
        anonymous_summary = copy.deepcopy(anonymous_summary)

        column_len = anonymous_summary["column_len"]
        anonymous_columns = [
            self._generate_anonymous_column(ANONYMOUS_COLUMN_PREFIX + str(i + column_len)) for i in range(len(columns))
        ]

        anonymous_summary["column_len"] = column_len + len(columns)
        return anonymous_columns, anonymous_summary

    def fill_anonymous_site_name(
        self, anonymous_label_name, anonymous_weight_name, anonymous_columns, anonymous_summary
    ):
        anonymous_summary = copy.deepcopy(anonymous_summary)

        self._check_site_name_consistency(anonymous_summary)

        if anonymous_summary["site_name"] is None:
            anonymous_label_name = self._fill_site_name(anonymous_label_name)
            anonymous_weight_name = self._fill_site_name(anonymous_weight_name)
            anonymous_columns = self._fill_site_name(anonymous_columns)
            anonymous_summary["site_name"] = self._site_name

        return dict(
            anonymous_label_name=anonymous_label_name,
            anonymous_weight_name=anonymous_weight_name,
            anonymous_columns=anonymous_columns,
            anonymous_summary=anonymous_summary,
        )

    def _fill_site_name(self, name):
        if name is None:
            return name

        if isinstance(name, str):
            site_name_pre, site_name_suf, suf = name.split(SPLICES, 2)
            site_name = SPLICES.join([site_name_pre, site_name_suf])

            if site_name != DEFAULT_SITE_NAME:
                raise ValueError(f"To fill anonymous names with site_name, it shouldn't be fill before")
            return self._generate_anonymous_column(suf)
        else:
            name = list(name)
            ret = []
            for _name in name:
                site_name_pre, site_name_suf, suf = _name.split(SPLICES, 2)
                site_name = SPLICES.join([site_name_pre, site_name_suf])

                if site_name != DEFAULT_SITE_NAME:
                    raise ValueError(f"To fill anonymous names with site_name, it shouldn't be fill before")

                ret.append(self._generate_anonymous_column(suf))

            return pd.Index(ret)
