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
from typing import List, Union
import pandas as pd


class Schema(object):
    def __init__(
        self,
        sample_id_name=None,
        match_id_name=None,
        weight_name=None,
        label_name=None,
        columns: Union[list, pd.Index] = None,
        anonymous_columns: Union[list, pd.Index] = None
    ):
        self._sample_id_name = sample_id_name
        self._match_id_name = match_id_name
        self._weight_name = weight_name
        self._label_name = label_name
        self._columns = pd.Index(columns) if columns else pd.Index([])
        self._anonymous_columns = pd.Index(anonymous_columns) if anonymous_columns else pd.Index([])

        self._name_index_mapping = dict()
        self._index_name_mapping = dict()

    @property
    def sample_id_name(self):
        return self._sample_id_name

    @sample_id_name.setter
    def sample_id_name(self, sample_id_name: str):
        self._sample_id_name = sample_id_name

    @property
    def match_id_name(self):
        return self._match_id_name

    @match_id_name.setter
    def match_id_name(self, match_id_name: str):
        self._match_id_name = match_id_name

    @property
    def weight_name(self):
        return self._weight_name

    @weight_name.setter
    def weight_name(self, weight_name: str):
        self._weight_name = weight_name

    @property
    def label_name(self):
        return self._label_name

    @label_name.setter
    def label_name(self, label_name: str):
        self._label_name = label_name

    @property
    def columns(self) -> pd.Index:
        return self._columns

    @columns.setter
    def columns(self, columns: pd.Index):
        self._columns = columns

    @property
    def anonymous_columns(self) -> pd.Index:
        return self._anonymous_columns

    def get_column_index(self, name):
        if name not in self._name_index_mapping:
            raise ValueError(f"{name} does not exist in schema")

        return self._name_index_mapping[name]

    def get_column_name(self, idx):
        if idx >= len(self._index_name_mapping):
            raise ValueError(f"Index={idx} is out out bound")

        return self._index_name_mapping[idx]

    def serialize(self):
        s_obj = list()
        s_obj.append(
            dict(name=self._sample_id_name,
                 property="sample_id")
        )

        if self._match_id_name:
            s_obj.append(
                dict(name=self._match_id_name,
                     property="match_id")
            )

        if self._label_name:
            s_obj.append(
                dict(name=self._label_name,
                     property="label")
            )
        if self._weight_name:
            s_obj.append(
                dict(name=self._weight_name,
                     property="weight")
            )

        if len(self._columns):
            if len(self._anonymous_columns):
                for name, anonymous_name in zip(self._columns, self._anonymous_columns):
                    s_obj.append(
                        dict(name=name,
                             anonymous_name=anonymous_name,
                             property="column")
                    )
            else:
                for name in self._columns:
                    s_obj.append(
                        dict(name=name,
                             property="column")
                    )

        return s_obj


class SchemaManager(object):
    def __init__(self):
        self._schema = None
        self._type_mapping = dict()
        self._name_index_mapping = dict()
        self._index_name_mapping = dict()

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, schema):
        self._schema = schema

    def get_all_keys(self):
        return list(self._name_index_mapping.keys())

    def parse_table_schema(self,
                           schema,
                           delimiter,
                           match_id_name,
                           label_name,
                           weight_name):
        sample_id_name = schema["sample_id_name"]
        columns = schema["header"].split(delimiter, -1)
        match_id_list = schema.get("match_id_list", [])

        return self.parse_local_file_schema(
            sample_id_name=sample_id_name,
            columns=columns,
            match_id_list=match_id_list,
            match_id_name=match_id_name,
            label_name=label_name,
            weight_name=weight_name
        )

    def parse_local_file_schema(self, sample_id_name, columns, match_id_list, match_id_name, label_name, weight_name):
        column_indexes = list(range(len(columns)))

        match_id_index, label_index, weight_index = None, None, None
        if match_id_list:
            if match_id_name and match_id_name not in match_id_list:
                raise ValueError(f"{match_id_name} not exist match_id_list={match_id_list}")
            if not match_id_name and len(match_id_list) > 1:
                raise ValueError(f"Multi match id exists, specify one to be used")

            match_id_name = match_id_list[0]
        elif match_id_name:
            match_id_list = [match_id_name]

        if match_id_name:
            match_id_index = self.extract_column_index_by_name(columns, column_indexes, match_id_name)
            match_id_list.pop(match_id_list.index(match_id_name))
        if label_name:
            label_index = self.extract_column_index_by_name(columns, column_indexes, label_name)
        if weight_name:
            weight_index = self.extract_column_index_by_name(columns, column_indexes, weight_name)

        for id_name in match_id_list:
            idx = columns.index(id_name)
            columns.pop(idx)
            column_indexes.pop(idx)

        self._schema = Schema(
            sample_id_name=sample_id_name,
            match_id_name=match_id_name,
            weight_name=weight_name,
            label_name=label_name,
            columns=columns
        )

        self.init_name_mapping()

        return dict(
            match_id_index=match_id_index,
            label_index=label_index,
            weight_index=weight_index,
            column_indexes=column_indexes
        )

    @staticmethod
    def extract_column_index_by_name(columns, column_indexes, name, drop=True):
        try:
            idx = columns.index(name)
            ret = column_indexes[idx]
            if drop:
                columns.pop(idx)
                column_indexes.pop(idx)

            return ret
        except ValueError:
            raise ValueError(f"{name} does not exist in {columns}")

    def init_column_types(self, label_type="float32", weight_type="float32", dtype="float32",
                          default_type="float32", match_id_type="index", sample_id_type="index"):
        self._type_mapping[self._schema.sample_id_name] = "index"

        if self._schema.match_id_name:
            self._type_mapping[self._schema.match_id_name] = "index"

        if self._schema.label_name:
            self._type_mapping[self._schema.label_name] = label_type

        if self._schema.weight_name:
            self._type_mapping[self._schema.weight_name] = weight_type

        if isinstance(dtype, str):
            for column in self._schema.columns:
                self._type_mapping[column] = dtype
        elif isinstance(dtype, dict):
            for column in self._schema.columns:
                self._type_mapping[column] = dtype.get(column, default_type)

    def init_name_mapping(self):
        offset = 1
        self._name_index_mapping[self._schema.sample_id_name] = 0

        if self._schema.match_id_name:
            self._name_index_mapping[self._schema.match_id_name] = offset
            offset += 1

        if self._schema.label_name:
            self._name_index_mapping[self._schema.label_name] = offset
            offset += 1

        if self._schema.weight_name:
            self._name_index_mapping[self._schema.weight_name] = offset
            offset += 1

        if len(self._schema.columns):
            for idx, column_name in enumerate(self._schema.columns):
                self._name_index_mapping[column_name] = offset + idx

        for column_name, idx in self._name_index_mapping.items():
            self._index_name_mapping[idx] = column_name

    def get_column_index(self, name):
        if name not in self._name_index_mapping:
            raise ValueError(f"{name} does not exist in schema")

        return self._name_index_mapping[name]

    def get_column_name(self, idx):
        if idx >= len(self._index_name_mapping):
            raise ValueError(f"Index={idx} is out out bound")

        return self._index_name_mapping[idx]

    def get_column_types(self, name=None, flatten=False):
        if not name:
            if not flatten:
                return self._type_mapping
            else:
                types = [None] * len(self._type_mapping)
                for idx, column_name in self._index_name_mapping:
                    types[idx] = self._type_mapping[column_name]
                return types
        else:
            return self._type_mapping[name]

    def derive_new_schema_manager(self, with_sample_id=True, with_match_id=True,
                                  with_label=True, with_weight=True, columns : Union[str, list] = None):
        derived_schema_manager = SchemaManager()
        derived_schema = Schema()

        indexes = []

        if with_sample_id:
            derived_schema.sample_id_name = self._schema.sample_id_name

        if with_match_id:
            derived_schema.match_id_name = self._schema.match_id_name

        if with_label:
            derived_schema.label_name = self._schema.label_name

        if with_weight:
            derived_schema.weight_name = self._schema.weight_name

        if columns:
            if isinstance(columns, str):
                columns = [columns]
            derived_schema.columns = self._schema.columns[self._schema.columns.get_indexer(columns)]

        derived_schema_manager.schema = derived_schema
        derived_schema_manager.init_name_mapping()

        for name in derived_schema_manager.get_all_keys():
            indexes.append(self.get_column_index(name))

        return derived_schema_manager, indexes

    def serialize(self):
        """
        fields: list of field,
                field:
                    name: column_name
                    anonymous_name: anonymous_column_name
                    dtype: data_type
                    property: sample_id/match_id/label/weight/column
        """
        schema_serialization = self._schema.serialize()

        schema_serialization_with_type = []
        for field in schema_serialization:
            field["dtype"] = self._type_mapping[field["name"]]
            schema_serialization_with_type.append(field)

        return schema_serialization_with_type

    @classmethod
    def deserialize(cls, fields: List[dict]):
        schema_manager = SchemaManager()

        schema_dict = dict(
            columns=[],
            anonymous_columns=[]
        )
        type_dict = {}
        for field in fields:
            p = field["property"]
            name = field["name"]
            if field["property"] in ["sample_id", "match_id", "weight", "label"]:
                schema_dict[f"{p}_name"] = name
                type_dict[f"{p}_type"] = field["dtype"]
            else:
                schema_dict["columns"].append(name)

                if "dtype" not in type_dict:
                    type_dict["dtype"] = dict()
                type_dict["dtype"][name] = field["dtype"]

                if "anonymous_name" in field:
                    schema_dict["anonymous_columns"].append(field["anonymous_name"])

        schema = Schema(**schema_dict)
        schema_manager.schema = schema
        schema_manager.init_name_mapping()
        schema_manager.init_column_types(**type_dict)

        return schema_manager
