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
from fate.arch.protocol.psi import psi_run
from fate.components.core import GUEST, HOST, Role, cpn


@cpn.component(roles=[GUEST, HOST], provider="fate")
def psi(
    ctx,
    role: Role,
    input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    protocol: cpn.parameter(type=str, default="ecdh_psi", optional=True),
    curve_type: cpn.parameter(type=str, default="curve25519", optional=True),
    output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    input_data = input_data.read()
    input_data_count = input_data.shape[0]
    intersect_data = psi_run(ctx, input_data, protocol, curve_type)
    summary = {
        "input_count": input_data_count,
        "intersect_count": intersect_data.shape[0],
        "intersect_rate": intersect_data.shape[0] / input_data_count,
        "method": "curve25519",
    }
    ctx.metrics.log_metrics(summary, "summary")
    output_data.write(intersect_data)
