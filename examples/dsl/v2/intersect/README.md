## Intersection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Intersection Task.

1. RAW Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_raw_conf.json

2. RAW Intersection with SM3 Hashing:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_raw_sm3_conf.json

3. RSA Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_conf.json

4. RSA Intersection with Random Base Fraction set to 0.5:
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_fraction_conf.json

5. RSA Intersection with Calculation Split:
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_split_conf.json

6. RSA Multi-hosts Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_multi_host_conf.json  
    
    This dsl is an example of guest runs intersection with two hosts using rsa intersection. It can be used as more than two hosts.
    
7. RAW Multi-hosts Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_raw_multi_host_conf.json  
    
    This dsl is an example of guest runs intersection with two hosts using rsa intersection. It can be used as more than two hosts.

8. DH Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_dh_conf.json
  
9. DH Multi-host Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_dh_multi_conf.json
    
10. RSA Intersection with Cache:  
    - dsl: test_intersect_job_cache_dsl.json  
    - runtime_config : test_intersect_job_rsa_cache_conf.json
    
11. DH Intersection with Cache:  
    - dsl: test_intersect_job_cache_dsl.json  
    - runtime_config : test_intersect_job_dh_cache_conf.json
   
   
12. RSA Intersection with Cache Loader:  
    - dsl: test_intersect_job_cache_loader_dsl.json  
    - runtime_config : test_intersect_job_rsa_cache_loader_conf.json
 
13. Intersect Cardinality:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_cardinality_conf.json
 
Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

Note: the intersection output only contains ids of intersection, because of the parameter of "only_output_key" in runtime_config.
