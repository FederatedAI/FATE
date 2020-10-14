## Intersection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Intersection Task.

1. Rsa Intersection:  
    dsl: test_intersect_job_dsl.json  
    runtime_config : test_intersect_job_rsa_conf.json

2. Raw Intersection:  
    dsl: test_intersect_job_dsl.json  
    runtime_config : test_intersect_job_raw_conf.json
    
3. Rsa Multi-hosts Intersection:  
    dsl: test_intersect_job_dsl.json  
    runtime_config : test_intersect_job_rsa_multi_host_conf.json  
    This dsl is an example of guest do intersection with two hosts using rsa intersection. It can be used as more than two hosts.
    
4. Raw Multi-hosts Intersection:  
    dsl: test_intersect_job_dsl.json  
    runtime_config : test_intersect_job_raw_multi_host_conf.json  
    This dsl is an example of guest do intersection with two hosts using rsa intersection. It can be used as more than two hosts.
    
Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

Note: the intersection output only contains ids of intersection, because of the parameter of "only_output_key" in runtime_config.
