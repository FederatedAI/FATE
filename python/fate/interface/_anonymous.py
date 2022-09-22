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
from typing import Protocol

ANONYMOUS_COLUMN_PREFIX = "x"
ANONYMOUS_LABEL = "y"
SPLICES = "_"


class Anonymous(Protocol):
    def migrate_schema_anonymous(self, schema):
        ...

    def migrate_anonymous(self, anonymous_header):
        ...

    def is_anonymous(self, column):
        ...

    def extend_columns(self, original_anonymous_header, extend_header):
        ...

    @staticmethod
    def get_party_id_from_anonymous_column(anonymous_column):
        ...

    @staticmethod
    def get_role_from_anonymous_column(anonymous_column):
        ...

    @staticmethod
    def get_suffix_from_anonymous_column(anonymous_column):
        ...

    @staticmethod
    def get_anonymous_header(schema):
        ...

    @staticmethod
    def filter_anonymous_header(schema, filter_ins):
        ...

    @staticmethod
    def reset_anonymous_header(schema, anonymous_header):
        ...

    @staticmethod
    def generate_derived_header(
        original_header, original_anonymous_header, derived_dict
    ):
        ...

    @staticmethod
    def generate_anonymous_column(role, party_id, suf):
        ...

    @staticmethod
    def get_anonymous_column_splits(column, num=-1):
        ...

    @staticmethod
    def is_expand_column(column_name):
        ...

    @staticmethod
    def get_expand_idx(column_name):
        ...

    @staticmethod
    def update_anonymous_header_with_role(schema, role, party_id):
        ...

    def generate_anonymous_header(self, schema):
        ...

    def generated_compatible_anonymous_header_with_old_version(self, header):
        ...

    @staticmethod
    def is_old_version_anonymous_header(anonymous_header):
        ...
