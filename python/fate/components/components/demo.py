import typing

from fate.components.core import ARBITER, GUEST, HOST, Role, cpn

if typing.TYPE_CHECKING:
    from fate.arch import Context


@cpn.component(roles=[GUEST, HOST, ARBITER])
def run(
    ctx: "Context",
    role: Role,
    parameter: cpn.parameter(type=str, default="default", desc="parameter"),
    # mix_input: cpn.dataframe_input(roles=[GUEST, HOST]) | cpn.data_directory_input(),
    mix_input: cpn.union(cpn.dataframe_input, cpn.data_directory_input)(roles=[GUEST, HOST]),
    dataframe_inputs: cpn.dataframe_inputs(roles=[GUEST, HOST]),
    # dataframe_input: cpn.dataframe_input(roles=[GUEST, HOST]),
    dataset_inputs: cpn.data_directory_inputs(roles=[GUEST, HOST]),
    dataset_input: cpn.data_directory_input(roles=[GUEST, HOST]),
    table_input: cpn.table_input(roles=[GUEST, HOST]),
    table_inputs: cpn.table_inputs(roles=[GUEST, HOST]),
    json_model_input: cpn.json_model_input(roles=[GUEST, HOST]),
    # dataframe_outputs: cpn.dataframe_outputs(roles=[GUEST, HOST]),
    # dataframe_output: cpn.dataframe_output(roles=[GUEST, HOST]),
    dataset_outputs: cpn.data_directory_outputs(roles=[GUEST, HOST]),
    dataset_output: cpn.data_directory_output(roles=[GUEST, HOST]),
    json_model_output: cpn.json_model_output(roles=[GUEST, HOST]),
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
    # json_model_output.write({"aaa": 1}, metadata={"bbb": 2})

    # output_path = model_directory_output.get_directory()
    # with open(output_path + "/a.txt", "w") as fw:
    #     fw.write("a")
    # model_directory_output.write_metadata({"model_directory_output_metadata": 1})

    ctx.metrics.log_accuracy("s", 1.0, 0)
    for i, sub_ctx in ctx.ctxs_range(10):
        sub_ctx.metrics.log_accuracy("sub", 1.0)
    print(ctx.metrics.handler._metrics)
    ctx.sub_ctx("ssss").metrics.log_loss("loss", 1.0, 0)

    ctx.metrics.log_metrics(
        [1, 2, 3, 4],
        "summary",
        "summary",
    )
    # print("dataframe_inputs", dataframe_inputs)

    ctx.metrics.log_accuracy("aaa", 1, 0)
