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
from fate.components import GUEST, HOST, Role
from fate.components.core import artifacts, component, parameter


@component(roles=[GUEST, HOST])
@parameter("path", type=str, default=None, optional=False)
@parameter("format", type=str, default="csv", optional=False)
@parameter("sample_id_name", type=str, default=None, optional=True)
@parameter("match_id_name", type=str, default=None, optional=True)
@parameter("delimiter", type=str, default=",", optional=True)
@parameter("label_name", type=str, default=None, optional=True)
@parameter("label_type", type=str, default="float32", optional=True)
@parameter("dtype", type=str, default="float32", optional=True)
@artifacts.dataframe_output("output_data", roles=[GUEST, HOST])
def reader(
    ctx,
    role: Role,
    path,
    format,
    sample_id_name,
    match_id_name,
    delimiter,
    label_name,
    label_type,
    dtype,
    output_data,
):
    read_data(ctx, path, format, sample_id_name, match_id_name, delimiter, label_name, label_type, dtype, output_data)


def read_data(ctx, path, format, sample_id_name, match_id_name, delimiter, label_name, label_type, dtype, output_data):
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
