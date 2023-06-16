from fate.components.core import ARBITER, GUEST, HOST, Role, cpn


@cpn.component(roles=[GUEST, HOST, ARBITER])
@cpn.dataframe_inputs("dataframe_inputs", roles=[GUEST, HOST])
@cpn.dataframe_input("dataframe_input", roles=[GUEST, HOST])
@cpn.table_input("table_input", roles=[GUEST, HOST])
@cpn.table_inputs("table_inputs", roles=[GUEST, HOST])
@cpn.data_directory_inputs("dataset_inputs", roles=[GUEST, HOST])
@cpn.data_directory_input("dataset_input", roles=[GUEST, HOST])
@cpn.dataframe_outputs("dataframe_outputs", roles=[GUEST, HOST])
@cpn.dataframe_output("dataframe_output", roles=[GUEST, HOST])
@cpn.data_directory_outputs("dataset_outputs", roles=[GUEST, HOST])
@cpn.data_directory_output("dataset_output", roles=[GUEST, HOST])
@cpn.json_model_input("json_model_input", roles=[GUEST, HOST])
@cpn.json_model_output("json_model_output", roles=[GUEST, HOST])
def run(
    ctx,
    role: Role,
    dataframe_inputs,
    dataframe_input,
    dataset_inputs,
    dataset_input,
    table_input,
    table_inputs,
    json_model_input,
    dataframe_outputs,
    dataframe_output,
    dataset_outputs,
    dataset_output,
    json_model_output,
):
    print("dataframe_inputs", dataframe_inputs)
    print("dataframe_input", dataframe_input)
    print("dataset_inputs", dataset_inputs)
    print("dataset_input", dataset_input)
    print("table_input", table_input)
    print("table_inputs", table_inputs)
    print("json_model_input", json_model_input)

    print("dataframe_outputs", dataframe_outputs)
    dataframe_outputs_0 = next(dataframe_outputs)
    dataframe_outputs_1 = next(dataframe_outputs)
    print("    dataframe_outputs_0", dataframe_outputs_0)
    print("    dataframe_outputs_1", dataframe_outputs_1)

    print("dataset_outputs", dataset_outputs)
    dataset_outputs_0 = next(dataset_outputs)
    dataset_outputs_1 = next(dataset_outputs)
    print("    dataset_outputs_0", dataset_outputs_0)
    print("    dataset_outputs_1", dataset_outputs_1)

    print("dataframe_output", dataframe_output)
    dataframe_output.write(ctx, dataframe_input, name="myname", namespace="mynamespace")
    print("dataset_output", dataset_output)

    json_model_output.write({"aaa": 1})
