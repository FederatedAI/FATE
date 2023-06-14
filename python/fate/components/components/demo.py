from fate.components import ARBITER, GUEST, HOST, Role
from fate.components.core import artifacts, cpn


@cpn.component(roles=[GUEST, HOST, ARBITER])
@artifacts.dataframe_inputs("dataframe_inputs", roles=[GUEST, HOST])
@artifacts.dataframe_input("dataframe_input", roles=[GUEST, HOST])
@artifacts.data_directory_inputs("dataset_inputs", roles=[GUEST, HOST])
@artifacts.data_directory_input("dataset_input", roles=[GUEST, HOST])
@artifacts.dataframe_outputs("dataframe_outputs", roles=[GUEST, HOST])
@artifacts.dataframe_output("dataframe_output", roles=[GUEST, HOST])
@artifacts.data_directory_outputs("dataset_outputs", roles=[GUEST, HOST])
@artifacts.data_directory_output("dataset_output", roles=[GUEST, HOST])
@artifacts.json_model_input("json_model_input", roles=[GUEST, HOST])
@artifacts.json_model_output("json_model_output", roles=[GUEST, HOST])
def run(
    ctx,
    role: Role,
    dataframe_inputs,
    dataframe_input,
    dataset_inputs,
    dataset_input,
    dataframe_outputs,
    dataframe_output,
    dataset_outputs,
    dataset_output,
    json_model_input,
    json_model_output,
):
    pass
