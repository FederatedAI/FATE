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
from fate.components.core import LOCAL, Role, cpn
from typing import Union, List, Dict


@cpn.component(roles=[LOCAL])
@cpn.table_input("table", roles=[LOCAL])
@cpn.dataframe_output("dataframe_output", roles=[LOCAL])
@cpn.parameter("namespace", type=str, default=",", optional=True)
@cpn.parameter("name", type=str, default="dense", optional=True)
@cpn.parameter("anonymous_role", type=str, default=None, optional=True)
@cpn.parameter("anonymous_party_id", type=str, default=None, optional=True)
def dataframe_transformer(
    ctx,
    role: Role,
    table,
    dataframe_output,
    namespace,
    name,
    anonymous_role,
    anonymous_party_id,
):
    from fate.arch.dataframe import TableReader
    metadata = table.schema
    table_reader = TableReader(
        sample_id_name=metadata.get("sample_id_name", None),
        match_id_name=metadata.get("match_id_name", None),
        match_id_list=metadata.get("match_id_list", None),
        match_id_range=metadata.get("match_id_range", 1),
        label_name=metadata.get("label_name", None),
        label_type=metadata.get("label_type", "int32"),
        weight_name=metadata.get("weight_name", None),
        weight_type=metadata.get("weight_type", "float32"),
        header=metadata.get("header", None),
        na_values=metadata.get("na_values", None),
        dtype=metadata.get("dtype", "float32"),
        anonymous_role=anonymous_role,
        anonymous_party_id=anonymous_party_id,
        delimiter=metadata.get("delimiter", ","),
        input_format=metadata.get("input_format", "dense"),
        tag_with_value=metadata.get("tag_with_value", False),
        tag_value_delimiter=metadata.get("tag_value_delimiter", ":")
    )

    df = table_reader.to_frame(ctx, table)
    ctx.writer(dataframe_output).write(df, namespace=namespace, name=name)
