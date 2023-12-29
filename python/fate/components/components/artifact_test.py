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

import typing

from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params

if typing.TYPE_CHECKING:
    from fate.arch import Context


@cpn.component(roles=[GUEST, HOST, ARBITER])
def artifact_test(
    ctx: "Context",
    role: Role,
    parameter: cpn.parameter(type=params.string_choice(["a", "b"]), desc="parameter", optional=False),
    # mix_input: cpn.dataframe_input(roles=[GUEST, HOST]) | cpn.data_directory_input(),
    mix_input: cpn.union(cpn.dataframe_input, cpn.data_directory_input)(roles=[GUEST, HOST]),
    dataframe_inputs: cpn.dataframe_inputs(roles=[GUEST, HOST]),
    # dataframe_input: cpn.dataframe_input(roles=[GUEST, HOST]),
    dataset_inputs: cpn.data_directory_inputs(roles=[GUEST, HOST]),
    dataset_input: cpn.data_directory_input(roles=[GUEST, HOST]),
    table_input: cpn.table_input(roles=[GUEST, HOST]),
    table_inputs: cpn.table_inputs(roles=[GUEST, HOST]),
    json_model_input: cpn.json_model_input(roles=[HOST]),
    # dataframe_outputs: cpn.dataframe_outputs(roles=[GUEST, HOST]),
    # dataframe_output: cpn.dataframe_output(roles=[GUEST, HOST]),
    dataset_outputs: cpn.data_directory_outputs(roles=[GUEST, HOST]),
    dataset_output: cpn.data_directory_output(roles=[GUEST, HOST]),
    json_model_output: cpn.json_model_output(roles=[GUEST, HOST]),
    json_model_outputs: cpn.json_model_outputs(roles=[GUEST, HOST]),
    model_directory_output: cpn.model_directory_output(roles=[GUEST, HOST]),
):
    # print("dataframe_input", dataframe_input)
    # print("dataset_inputs", dataset_inputs)
    # print("dataset_input", dataset_input)
    # print("table_input", table_input)
    # print("table_inputs", table_inputs)
    # print("json_model_input", json_model_input)
    #
    # print("dataframe_outputs", dataframe_outputs)
    # dataframe_outputs_0 = next(dataframe_outputs)
    # dataframe_outputs_1 = next(dataframe_outputs)
    # print("    dataframe_outputs_0", dataframe_outputs_0)
    # print("    dataframe_outputs_1", dataframe_outputs_1)
    #
    print("dataset_outputs", dataset_outputs)
    dataset_outputs_0 = next(dataset_outputs)
    dataset_outputs_1 = next(dataset_outputs)
    print("    dataset_outputs_0", dataset_outputs_0)
    print("    dataset_outputs_1", dataset_outputs_1)
    #
    # print("dataframe_output", dataframe_output)
    # dataframe_output.write(dataframe_input.read(), name="myname", namespace="mynamespace")
    # print("dataset_output", dataset_output)
    #
    next(json_model_outputs).write({"aaa": 1}, metadata={"bbb": 2})
    next(json_model_outputs).write({"aaa": 2}, metadata={"bbb": 2})

    json_model_output.write({"aaa": 1}, metadata={"bbb": 2})

    model_directory_output.get_directory()

    # output_path = model_directory_output.get_directory()
    # with open(output_path + "/a.txt", "w") as fw:
    #     fw.write("a")
    # model_directory_output.write_metadata({"model_directory_output_metadata": 1})

    # ctx.metrics.log_accuracy("s", 1.0, 0)
    # for i, sub_ctx in ctx.ctxs_range(10):
    #     sub_ctx.metrics.log_accuracy("sub", 1.0)
    # print(ctx.metrics.handler._metrics)
    # ctx.sub_ctx("ssss").metrics.log_loss("loss", 1.0, 0)
    #
    # ctx.metrics.log_metrics(
    #     [1, 2, 3, 4],
    #     "summary",
    #     "summary",
    # )
    # # print("dataframe_inputs", dataframe_inputs)
    #
    # ctx.metrics.log_accuracy("aaa", 1, 0)
    ctx.metrics.log_loss("loss_aa", [1.0, 2.0], 0)
    ctx.metrics.log_metrics(values=[1, 2, 3], name="metric_single", type="custom", metadata={"bbb": 2})
    ctx.sub_ctx("sub_metric").metrics.log_loss("loss", 1.0, 0)
    ctx.sub_ctx("sub_metric").metrics.log_loss("loss", 0.9, 1)

    for i, auto_step_sub_ctx in ctx.ctxs_range(10):
        auto_step_sub_ctx.metrics.log_accuracy("sub", 1.0)
