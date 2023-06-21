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


@cpn.component(roles=[LOCAL])
@cpn.dataframe_input("dataframe_input", roles=[LOCAL])
@cpn.dataframe_inputs("dataframe_inputs", roles=[LOCAL])
@cpn.dataframe_output("dataframe_output", roles=[LOCAL])
@cpn.dataframe_outputs("dataframe_outputs", roles=[LOCAL])
@cpn.json_model_output("json_model_output", roles=[LOCAL])
@cpn.json_model_outputs("json_model_outputs", roles=[LOCAL])
def dataframe_io_test(
    ctx,
    role: Role,
    dataframe_input,
    dataframe_output,
    dataframe_inputs,
    dataframe_outputs,
    json_model_output,
    json_model_outputs,
):
    dataframe_input = dataframe_input + 1

    dataframe_output.write(ctx, dataframe_input)

    assert len(dataframe_inputs) == 4
    for i in range(10):
        output = next(dataframe_outputs)
        output.write(ctx, dataframe_inputs[i % 4])

    json_model_output.write({"aaa": 1}, metadata={"bbb": 2})
    for i in range(8):
        model_output = next(json_model_outputs)
        model_output.write({"io_model_out": i}, metadata={"i-th output": i})
