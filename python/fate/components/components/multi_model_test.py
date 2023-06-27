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
from fate.arch import Context
from fate.components.core import LOCAL, Role, cpn


@cpn.component(roles=[LOCAL])
def multi_model_test(
    ctx: Context,
    role: Role,
    dataframe_input: cpn.dataframe_input(roles=[LOCAL]),
    dataframe_output: cpn.dataframe_output(roles=[LOCAL]),
    json_model_input: cpn.json_model_input(roles=[LOCAL]),
    json_model_inputs: cpn.json_model_inputs(roles=[LOCAL]),
    json_model_output: cpn.json_model_output(roles=[LOCAL]),
    json_model_outputs: cpn.json_model_outputs(roles=[LOCAL]),
    model_directory_output: cpn.model_directory_output(roles=[LOCAL]),
    model_directory_outputs: cpn.model_directory_outputs(roles=[LOCAL]),
):
    df = dataframe_input.read()
    df = df + 1

    dataframe_output.write(df)
    json_model_input = json_model_input.read()
    json_model_inputs = [model_input.read() for model_input in json_model_inputs]

    assert json_model_input == {"aaa": 1}
    assert len(json_model_inputs) == 3
    json_model_output.write({"ccccccc": 1}, metadata={"ccc": 2})
    for i in range(10):
        model_output = next(json_model_outputs)
        model_output.write({"abc": i}, metadata={"i-th output": i})

    output_path = model_directory_output.get_directory()
    with open(output_path + "/a.txt", "w") as fw:
        fw.write("test for model directory output\n")

    model_directory_output.write_metadata({"model_directory_output_metadata": "test_directory"})

    for i in range(5):
        directory_output = next(model_directory_outputs)
        path = directory_output.get_directory()
        with open(path + f"/output_{i}.txt", "w") as fw:
            fw.write("test for model directory output\n")

        directory_output.write_metadata({f"model_directory_output_{i}_metadata": f"test_directory_{i}"})

    ctx.metrics.log_metrics(values=[1, 2, 3], name="metric_single", type="custom", metadata={"bbb": 2})
    ctx.sub_ctx("sub_metric").metrics.log_loss("loss", 1.0, 0)
    ctx.sub_ctx("sub_metric").metrics.log_loss("loss", 0.9, 1)

    for i, auto_step_sub_ctx in ctx.ctxs_range(10):
        auto_step_sub_ctx.metrics.log_accuracy("sub", 1.0)
