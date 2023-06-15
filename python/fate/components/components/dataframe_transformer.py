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
@cpn.parameter("sample_id_name", type=str, default=None, optional=True)
@cpn.parameter("match_id_name", type=str, default=None, optional=True)
@cpn.parameter("match_id_list", type=List[str], default=None, optional=True)
@cpn.parameter("match_id_range", type=int, default=1, optional=True)
@cpn.parameter("label_name", type=str, default=None, optional=True)
@cpn.parameter("label_type", type=str, default="int32", optional=True)
@cpn.parameter("weight_name", type=str, default=None, optional=True)
@cpn.parameter("weight_type", type=str, default="float32", optional=True)
@cpn.parameter("header", type=str, default=None, optional=False)
@cpn.parameter("na_values", type=Union[str, list, dict], default=None, optional=True)
@cpn.parameter("dtype", type=Union[Dict, str], default="float32", optional=True)
@cpn.parameter("anonymous_role", type=str, default=None, optional=True)
@cpn.parameter("anonymous_party_id", type=str, default=None, optional=True)
@cpn.parameter("delimiter", type=str, default=",", optional=True)
@cpn.parameter("input_format", type=str, default="dense", optional=True)
@cpn.parameter("tag_with_value", type=bool, default=False, optional=True)
@cpn.parameter("tag_value_delimiter", type=str, default=":", optional=True)
def dataframe_transformer(
    ctx,
    role: Role,
    table,
    dataframe_output,
    sample_id_name,
    match_id_name,
    match_id_list,
    match_id_range,
    label_name,
    label_type,
    weight_name,
    weight_type,
    header,
    na_values,
    dtype,
    anonymous_role,
    anonymous_party_id,
    delimiter,
    input_format,
    tag_with_value,
    tag_value_delimiter
):
    from fate.arch.dataframe import TableReader

    table_reader = TableReader(
        sample_id_name=sample_id_name,
        match_id_name=match_id_name,
        match_id_list=match_id_list,
        match_id_range=match_id_range,
        label_name=label_name,
        label_type=label_type,
        weight_name=weight_name,
        weight_type=weight_type,
        header=header,
        na_values=na_values,
        dtype=dtype,
        anonymous_role=anonymous_role,
        anonymous_party_id=anonymous_party_id,
        delimiter=delimiter,
        input_format=input_format,
        tag_with_value=tag_with_value,
        tag_value_delimiter=tag_value_delimiter
    )

    df = table_reader.to_frame(ctx, table)
    ctx.writer(dataframe_output).write_dataframe(df)
