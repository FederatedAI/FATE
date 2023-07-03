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


@cpn.component(roles=[GUEST, HOST], provider="fate")
def intersection(
    ctx,
    role: Role,
    input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    method: cpn.parameter(type=str, default="raw", optional=True),
    output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    if role.is_guest:
        if method == "raw":
            raw_intersect_guest(ctx, input_data, output_data)
    elif role.is_host:
        if method == "raw":
            raw_intersect_host(ctx, input_data, output_data)


def raw_intersect_guest(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionGuest

    data = input_data.read()
    # data = ctx.reader(input_data).read_dataframe().data
    guest_intersect_obj = RawIntersectionGuest()
    intersect_data = guest_intersect_obj.fit(ctx, data)
    # ctx.writer(output_data).write_dataframe(intersect_data)
    output_data.write(intersect_data)


def raw_intersect_host(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionHost

    data = input_data.read()
    # data = ctx.reader(input_data).read_dataframe().data
    host_intersect_obj = RawIntersectionHost()
    intersect_data = host_intersect_obj.fit(ctx, data)
    # ctx.writer(output_data).write_dataframe(intersect_data)
    output_data.write(intersect_data)
