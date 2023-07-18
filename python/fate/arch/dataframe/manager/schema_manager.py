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
from typing import List, Union
import pandas as pd


DEFAULT_LABEL_NAME = "label"
DEFAULT_WEIGHT_NAME = "weight"


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

    def append_columns(self, names):
        self._columns = self._columns.append(pd.Index(names))
        # TODO: extend anonymous column

    def pop_columns(self, names):
        names = set(names)
        if self._label_name in names:
            names.remove(self._label_name)
            self._label_name = None
        if self._weight_name  in names:
            names.remove(self._weight_name)
            self._weight_name = None

        columns = []
        for name in self._columns:
            if name not in names:
                columns.append(name)
        self._columns = pd.Index(columns)

        # TODO: pop anonymous columns

    def __eq__(self, other: "Schema"):
        return self.label_name == other.label_name and self.weight_name == other.weight_name \
               and self.sample_id_name == other.sample_id_name and self.match_id_name == other.match_id_name \
               and self.columns.tolist() == other.columns.tolist()

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
        self._name_offset_mapping = dict()
        self._offset_name_mapping = dict()

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, schema):
        self._schema = schema

    def rename(self, sample_id_name=None, match_id_name=None, label_name=None, weight_name=None, columns: dict = None):
        attr_dict = {
            "sample_id_name": sample_id_name,
            "match_id_name": match_id_name,
            "label_name": label_name,
            "weight_name": weight_name
        }

        for attr, value in attr_dict.items():
            if not value:
                continue
            o_name = getattr(self._schema, attr)
            setattr(self._schema, attr, value)
            self._rename_single_column(o_name, value)

        if columns:
            for o_name, n_name in columns.items():
                self._rename_single_column(o_name, n_name)

            o_columns = self._schema.columns.tolist()
            n_columns = [o_name if o_name not in columns else columns[o_name] for o_name in o_columns]
            self._schema.columns = pd.Index(n_columns)

    def _rename_single_column(self, src, dst):
        self._type_mapping[dst] = self._type_mapping[src]
        self._type_mapping.pop(src)

        self._name_offset_mapping[dst] = self._name_offset_mapping[src]
        offset = self._name_offset_mapping.pop(src)

        self._offset_name_mapping[offset] = dst

    def add_label_or_weight(self, key_type, name, block_type):
        self._type_mapping[name] = block_type.value

        src_field_names = self.get_field_name_list()
        if key_type == "label":
            self._schema.label_name = name
        else:
            self._schema.weight_name = name

        dst_field_names = self.get_field_name_list()

        name_offset_mapping = dict()
        offset_name_mapping = dict()
        field_index_changes = dict()

        for offset, field_name in enumerate(dst_field_names):
            name_offset_mapping[field_name] = offset
            offset_name_mapping[offset] = field_name

        for field_name in src_field_names:
            src_offset = self._name_offset_mapping[field_name]
            dst_offset = name_offset_mapping[field_name]
            field_index_changes[src_offset] = dst_offset

        self._name_offset_mapping = name_offset_mapping
        self._offset_name_mapping = offset_name_mapping

        return self._name_offset_mapping[name], field_index_changes

    def append_columns(self, names, block_types):
        field_index = len(self._name_offset_mapping)
        for offset, name in enumerate(names):
            if isinstance(block_types, list):
                dtype = block_types[offset].value
            else:
                dtype = block_types.value

            self._type_mapping[name] = dtype
            self._name_offset_mapping[name] = field_index + offset
            self._offset_name_mapping[field_index + offset] = name

        self.schema.append_columns(names)

        return [field_index + offset for offset in range(len(names))]

    def pop_fields(self, field_indexes):
        field_names = [self._offset_name_mapping[field_id] for field_id in field_indexes]
        self._schema = copy.deepcopy(self._schema)
        self._schema.pop_columns(field_names)

        field_index_set = set(field_indexes)
        left_field_indexes = []
        for i in range(len(self._offset_name_mapping)):
            if i not in field_index_set:
                left_field_indexes.append(i)

        name_offset_mapping = dict()
        offset_name_mapping = dict()
        field_index_changes = dict()
        for dst_field_id, src_field_id in enumerate(left_field_indexes):
            name = self._offset_name_mapping[src_field_id]
            name_offset_mapping[name] = dst_field_id
            offset_name_mapping[dst_field_id] = name
            field_index_changes[src_field_id] = dst_field_id

        self._name_offset_mapping = name_offset_mapping
        self._offset_name_mapping = offset_name_mapping

        return field_index_changes

    def split_columns(self, names, block_types):
        field_indexes = [self._name_offset_mapping[name] for name in names]
        for offset, name in enumerate(names):
            if isinstance(block_types, list):
                self._type_mapping[name] = block_types[offset].value
            else:
                self._type_mapping[name] = block_types.value

        return field_indexes

    def duplicate(self):
        dup_schema_manager = SchemaManager()
        dup_schema_manager.schema = copy.deepcopy(self._schema)
        dup_schema_manager._name_offset_mapping = copy.deepcopy(self._name_offset_mapping)
        dup_schema_manager._type_mapping = copy.deepcopy(self._type_mapping)
        dup_schema_manager._offset_name_mapping = copy.deepcopy(self._offset_name_mapping)

        return dup_schema_manager

    def get_all_keys(self):
        return list(self._name_offset_mapping.keys())

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

    def init_field_types(self, label_type="float32", weight_type="float32", dtype="float32",
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
        self._name_offset_mapping[self._schema.sample_id_name] = 0

        if self._schema.match_id_name:
            self._name_offset_mapping[self._schema.match_id_name] = offset
            offset += 1

        if self._schema.label_name:
            self._name_offset_mapping[self._schema.label_name] = offset
            offset += 1

        if self._schema.weight_name:
            self._name_offset_mapping[self._schema.weight_name] = offset
            offset += 1

        if len(self._schema.columns):
            for idx, column_name in enumerate(self._schema.columns):
                self._name_offset_mapping[column_name] = offset + idx

        for column_name, idx in self._name_offset_mapping.items():
            self._offset_name_mapping[idx] = column_name

    def get_field_offset(self, name):
        if name not in self._name_offset_mapping:
            raise ValueError(f"{name} does not exist in schema")

        return self._name_offset_mapping[name]

    def get_field_name(self, offset):
        if offset >= len(self._offset_name_mapping):
            raise ValueError(f"Offset={offset} is out out bound")

        return self._offset_name_mapping[offset]

    def get_field_name_list(self, with_sample_id=True, with_match_id=True, with_label=True, with_weight=True):
        field_names = []
        if with_sample_id and self._schema.sample_id_name:
            field_names.append(self._schema.sample_id_name)

        if with_match_id and self._schema.match_id_name:
            field_names.append(self._schema.match_id_name)

        if with_label and self._schema.label_name:
            field_names.append(self._schema.label_name)

        if with_weight and self._schema.weight_name:
            field_names.append(self._schema.weight_name)

        field_names += self._schema.columns.tolist()

        return field_names

    def get_field_types(self, name=None, flatten=False):
        if not name:
            if not flatten:
                return self._type_mapping
            else:
                types = [None] * len(self._type_mapping)
                for idx, name in self._offset_name_mapping:
                    types[idx] = self._type_mapping[name]
                return types
        else:
            return self._type_mapping[name]

    def set_field_type_by_offset(self, field_index, field_type):
        name = self._offset_name_mapping[field_index]
        self._type_mapping[name] = field_type

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
            indexes.append((self.get_field_offset(name), derived_schema_manager.get_field_offset(name)))
            derived_schema_manager._type_mapping[name] = self.get_field_types(name)

        return derived_schema_manager, sorted(indexes)

    def infer_operable_field_names(self) -> List[str]:
        if len(self._schema.columns):
            return self._schema.columns.tolist()
        elif self._schema.weight_name:
            return [self._schema.weight_name]
        else:
            return [self._schema.label_name]

    def infer_operable_filed_offsets(self) -> List[int]:
        operable_field_names = self.infer_operable_field_names()
        operable_field_offsets = [self._name_offset_mapping[field_name] for field_name in operable_field_names]

        return operable_field_offsets

    def infer_non_operable_field_names(self) -> List[str]:
        operable_field_name_set = set(self.infer_operable_field_names())
        non_operable_field_names = []
        if self._schema.sample_id_name:
            non_operable_field_names.append(self._schema.sample_id_name)

        if self._schema.match_id_name:
            non_operable_field_names.append(self._schema.match_id_name)

        if self._schema.label_name and self._schema.label_name not in operable_field_name_set:
            non_operable_field_names.append(self._schema.label_name)

        if self._schema.weight_name and self._schema.weight_name not in operable_field_name_set:
            non_operable_field_names.append(self._schema.weight_name)

        return non_operable_field_names

    def infer_non_operable_field_offsets(self) -> List[int]:
        non_operable_field_names = self.infer_non_operable_field_names()
        non_operable_field_offsets = [self._name_offset_mapping[field_name] for field_name in non_operable_field_names]

        return non_operable_field_offsets

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
        schema_manager.init_field_types(**type_dict)

        return schema_manager
