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
from fate.components import GUEST, HOST, DatasetArtifact, Input, Output, Role, cpn


@cpn.component(roles=[GUEST, HOST], provider="fate")
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("method", type=str, default="raw", optional=True)
@cpn.artifact("output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def intersection(
    ctx,
    role: Role,
    input_data,
    method,
    output_data,
):
    if role.is_guest:
        if method == "raw":
            raw_intersect_guest(ctx, input_data, output_data)
    elif role.is_host:
        if method == "raw":
            raw_intersect_host(ctx, input_data, output_data)


def raw_intersect_guest(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionGuest

    data = ctx.reader(input_data).read_dataframe().data
    guest_intersect_obj = RawIntersectionGuest()
    intersect_data = guest_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)


def raw_intersect_host(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionHost

    data = ctx.reader(input_data).read_dataframe().data
    host_intersect_obj = RawIntersectionHost()
    intersect_data = host_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)
