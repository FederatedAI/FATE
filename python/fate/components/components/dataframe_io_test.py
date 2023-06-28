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
def dataframe_io_test(
    ctx,
    role: Role,
    dataframe_input: cpn.dataframe_input(roles=[LOCAL]),
    dataframe_output: cpn.dataframe_output(roles=[LOCAL]),
    dataframe_inputs: cpn.dataframe_inputs(roles=[LOCAL]),
    dataframe_outputs: cpn.dataframe_outputs(roles=[LOCAL]),
    json_model_output: cpn.json_model_output(roles=[LOCAL]),
    json_model_outputs: cpn.json_model_outputs(roles=[LOCAL]),
    model_directory_output: cpn.model_directory_output(roles=[LOCAL]),
    model_directory_outputs: cpn.model_directory_outputs(roles=[LOCAL]),
):
    df = dataframe_input.read()
    df = df + 1
    df_list = [_input.read() for _input in dataframe_inputs]
    dataframe_output.write(df)

    assert len(df_list) == 4
    for i in range(10):
        output = next(dataframe_outputs)
        output.write(df_list[i % 4])

    json_model_output.write({"aaa": 1}, metadata={"bbb": 2})
    for i in range(8):
        model_output = next(json_model_outputs)
        model_output.write({"io_model_out": i}, metadata={"i-th output": i})

    path = model_directory_output.get_directory()
    with open(path + "/output.txt", "w") as fw:
        fw.write("xxx\n")

    model_directory_output.write_metadata({"model_directory_output_metadata": "test_directory"})

    for i in range(5):
        directory_output = next(model_directory_outputs)
        path = directory_output.get_directory()
        with open(path + f"/output_{i}.txt", "w") as fw:
            fw.write("test for model directory output\n")

        directory_output.write_metadata({f"model_directory_output_{i}_metadata": f"test_directory_{i}"})
