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
@cpn.dataframe_output("dataframe_output", roles=[LOCAL])
@cpn.json_model_input("json_model_input", roles=[LOCAL])
@cpn.json_model_inputs("json_model_inputs", roles=[LOCAL])
@cpn.json_model_output("json_model_output", roles=[LOCAL])
@cpn.json_model_outputs("json_model_outputs", roles=[LOCAL])
@cpn.model_directory_output("model_directory_output", roles=[LOCAL])
@cpn.model_directory_outputs("model_directory_outputs", roles=[LOCAL])
def multi_model_test(
    ctx,
    role: Role,
    dataframe_input,
    dataframe_output,
    json_model_input,
    json_model_inputs,
    json_model_output,
    json_model_outputs,
    model_directory_output,
    model_directory_outputs,
):
    dataframe_input = dataframe_input + 1

    dataframe_output.write(ctx, dataframe_input)

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
