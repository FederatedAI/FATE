## Data Split Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.


#### Tasks.

1. Heterogeneous Data Split Task:
    dsl: test_hetero_data_split_job_dsl.json
    runtime_config : test_hetero_data_split_job_conf.json
    data type: continuous
    stratification: stratified by given split points

2. Homogeneous Data Spilt Task:
    dsl: test_homo_data_split_job_dsl.json
    runtime_config: test_homo_data_split_job_conf.json
    data type: categorical
    stratification: stratified by label

Users can use following commands to run a task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}
