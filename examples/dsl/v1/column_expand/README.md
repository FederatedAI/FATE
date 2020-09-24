## Column Expand Configuration Usage Guide.

This section introduces the dsl and conf for usage of different types of tasks.


#### Tasks.

1. Column Expand Task:
    dsl: test_column_expand_job_dsl.json
    runtime_config : test_column_expand_job_conf.json
    data type: categorical(binary)

Users can use following commands to run a task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}
