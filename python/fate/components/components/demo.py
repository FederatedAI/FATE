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
    dataframe_outputs,
    dataframe_output,
    dataset_outputs,
    dataset_output,
    json_model_input,
    json_model_output,
):
    json_model_output.write({"aaa": 1})
