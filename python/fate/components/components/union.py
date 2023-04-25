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

from fate.components import (
    GUEST,
    HOST,
    DatasetsArtifacts,
    DatasetArtifact,
    Input,
    Output,
    Role,
    cpn,
    params
)


@cpn.component(roles=[GUEST, HOST])
def union(ctx, role):
    ...


@union.train()
@cpn.artifact("train_data_list", type=Input[DatasetsArtifacts], roles=[GUEST, HOST])
@cpn.parameter("axis", type=params.conint(strict=True, ge=0, le=1), default=0, optional=False,
               desc="axis along which concatenation is performed, 0 for row-wise, 1 for column-wise")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def union_train(
        ctx,
        role: Role,
        train_data_list,
        method,
        train_output_data
):
    train(ctx, train_data_list, train_output_data, method)


def train(ctx, train_data_list, train_output_data, method):
    from fate.ml.preprocessing import Union
    data_list = []
    for data in train_data_list:
        data = ctx.reader(data).read_dataframe().data
        data_list.append(data)

    with ctx.sub_ctx("train") as sub_ctx:
        union_obj = Union(method)
        output_data = union_obj.fit(sub_ctx, data_list)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)
