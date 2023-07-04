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
from fate.components.core import GUEST, HOST, Role, cpn


@cpn.component(roles=[GUEST, HOST])
def reader(
    ctx,
    role: Role,
    path: cpn.parameter(type=str, default=None, optional=False),
    format: cpn.parameter(type=str, default="csv", optional=False),
    sample_id_name: cpn.parameter(type=str, default=None, optional=True),
    match_id_name: cpn.parameter(type=str, default=None, optional=True),
    delimiter: cpn.parameter(type=str, default=",", optional=True),
    label_name: cpn.parameter(type=str, default=None, optional=True),
    label_type: cpn.parameter(type=str, default="float32", optional=True),
    dtype: cpn.parameter(type=str, default="float32", optional=True),
    output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    if format == "csv":
        data_meta = DataframeArtifact(
            uri=path,
            name="data",
            metadata=dict(
                format=format,
                sample_id_name=sample_id_name,
                match_id_name=match_id_name,
                delimiter=delimiter,
                label_name=label_name,
                label_type=label_type,
                dtype=dtype,
            ),
        )
    elif format == "raw_table":
        data_meta = DataframeArtifact(uri=path, name="data", metadata=dict(format=format))
    else:
        raise ValueError(f"Reader does not support format={format}")

    data = ctx.reader(data_meta).read_dataframe()
    ctx.writer(output_data).write_dataframe(data)
