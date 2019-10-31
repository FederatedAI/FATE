## Intersection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Intersection Task.

1. Rsa intersection:
    dsl: test_intersect_job_dsl.json
    runtime_config : test_intersect_job_rsa_conf.json

2. Raw intersection:
    dsl: test_intersect_job_dsl.json
    runtime_config : test_intersect_job_raw_conf.json
    
3. Rsa multi-hosts intersection:
    dsl: test_intersect_job_dsl.json
    runtime_config : test_intersect_job_rsa_multi_host_conf.json
    This dsl is an example of guest do intersection with two hosts using rsa intersection. It can be used as more than two hosts, as well as raw intersection.
    
4. Rsa intersection using cache to speed up:
    dsl: test_intersect_job_dsl.json
    runtime_config: test_intersect_job_rsa_using_cache.json
    
Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Note: the intersection output only contains ids of intersection, because of the parameter of "only_output_key" in runtime_config.
