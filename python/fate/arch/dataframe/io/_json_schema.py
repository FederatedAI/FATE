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
import pandas as pd

from ..storage import ValueStore

FRAME_SCHEME = "fate.dataframe"


def build_schema(data):
    fields = []
    schema = data.schema
    """
    index, match_id, label, weight, values
    """
    fields.append(dict(type="str", name=schema.sid, property="index"))

    if schema.match_id_name is not None:
        fields.append(dict(type="str", name=schema.match_id_name, property="match_id"))

    if schema.label_name is not None:
        label = data.label
        fields.append(dict(type=label.dtype.name, name=schema.label_name, property="label"))

    if schema.weight_name is not None:
        weight = data.weight
        fields.append(dict(type=weight.dtype.name, name=schema["weight_name"], property="weight"))

    if schema.header is not None:
        values = data.values
        columns = schema.header
        if isinstance(values, ValueStore):
            dtypes = values.dtypes
            for col_name in columns:
                fields.append(
                    dict(
                        type=dtypes[col_name].name,
                        name=col_name,
                        property="value",
                        source="fate.dataframe.value_store",
                    )
                )

        else:
            for col_name in columns:
                fields.append(dict(type=values.dtype.name, name=col_name, property="value", source="fate.arch.tensor"))

    built_schema = dict()
    built_schema["fields"] = fields
    built_schema["global_ranks"] = data.index.global_ranks
    built_schema["block_partition_mapping"] = data.index.block_partition_mapping
    built_schema["type"] = FRAME_SCHEME
    return built_schema


def parse_schema(schema):
    if "type" not in schema or schema["type"] != FRAME_SCHEME:
        raise ValueError(f"deserialize data error, schema type is not {FRAME_SCHEME}")

    recovery_schema = dict()
    column_info = dict()
    fields = schema["fields"]

    for idx, field in enumerate(fields):
        if field["property"] == "index":
            recovery_schema["sid"] = field["name"]
            column_info["index"] = dict(start_idx=idx, end_idx=idx, type=field["type"])

        elif field["property"] == "match_id":
            recovery_schema["match_id_name"] = field["name"]
            column_info["match_id"] = dict(start_idx=idx, end_idx=idx, type=field["type"])

        elif field["property"] == "label":
            recovery_schema["label_name"] = field["name"]
            column_info["label"] = dict(start_idx=idx, end_idx=idx, type=field["type"])

        elif field["property"] == "weight":
            recovery_schema["weight_name"] = field["name"]
            column_info["weight"] = dict(start_idx=idx, end_idx=idx, type=field["type"])

        elif field["property"] == "value":
            header = [field["name"] for field in fields[idx:]]
            recovery_schema["header"] = header
            column_info["values"] = dict(
                start_idx=idx, end_idx=idx + len(header) - 1, type=field["type"], source=field["source"]
            )
            break

    return recovery_schema, schema["global_ranks"], schema["block_partition_mapping"], column_info
