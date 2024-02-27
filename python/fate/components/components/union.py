#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn


@cpn.component(roles=[GUEST, HOST], provider="fate")
def union(
    ctx: Context,
    role: Role,
    input_datas: cpn.dataframe_inputs(roles=[GUEST, HOST]),
    output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    from fate.ml.preprocessing import Union

    data_list = []
    data_len_dict = {}
    for data in input_datas:
        data_name = f"{data.artifact.metadata.source.task_name}.{data.artifact.metadata.source.unique_key()}"
        # logger.debug(f"data_name: {data_name}")
        data = data.read()
        data_list.append(data)
        data_len_dict[data_name] = data.shape[0]

    sub_ctx = ctx.sub_ctx("train")
    union_obj = Union()
    output_df = union_obj.fit(sub_ctx, data_list)
    data_len_dict["total"] = output_df.shape[0]
    ctx.metrics.log_metrics(data_len_dict, "summary")
    output_data.write(output_df)
