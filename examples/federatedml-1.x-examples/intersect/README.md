## Intersection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Intersection Task.

1. Rsa intersection:
    dsl: test_intersect_job_dsl.json
    runtime_config : test_intersect_job_rsa_conf.json

2. Raw intersection:
    dsl: test_intersect_job_dsl.json
    runtime_config : test_intersect_job_raw_conf.json

Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Note: the intersection output is only ids of intersection, because of the parameter of "only_output_key" in runtime_config.  
