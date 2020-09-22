## Data Split Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

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

3. Heterogeneous Data Split Task with Multiple Models:
    dsl: test_hetero_data_split_multi_model_job_dsl.json
    runtime_config: test_hetero_data_split_multi_model_job_conf.json
    data type: categorical
    stratification: stratified by label

Users can use following commands to run a task.

    bash flow job submit -c ${runtime_config} -d ${dsl}
