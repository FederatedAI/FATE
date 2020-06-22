## Union Configuration Usage Guide.

This section introduces the dsl and conf for usage of different tasks.

1. Unilateral Union Task:

    dsl: test_union_basic_job_dsl.json

    runtime_config : test_union_basic_job_conf.json

2.  Union Task on Both Guest & Host Sides:

    dsl: test_union_job_dsl.json

    runtime_config : test_union_job_conf.json

3. Union Task on Table:

    dsl: test_union_dataio_job_dsl.json
    
    runtime_config: test_union_basic_job_conf.json
    
4. Union Task on TagValue Table (with duplicated ids):
    
    dsl: test_union_dataio_job_dsl.json
    
    runtime_config: test_union_tag_value_job_conf.json

Users can use following commands to run the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check output. 