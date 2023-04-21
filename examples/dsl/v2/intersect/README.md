## Intersection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Intersection Task.

1. RSA Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_conf.json

2. RSA Intersection with Random Base Fraction set to 0.5:
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_fraction_conf.json

3. RSA Intersection with Calculation Split:
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_split_conf.json

4. RSA Multi-hosts Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_multi_host_conf.json  
    
    This dsl is an example of guest runs intersection with two hosts using rsa intersection. It can be used as more than two hosts.
    
5. DH Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_dh_conf.json
  
6. DH Multi-host Intersection:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_dh_multi_conf.json
    
7. ECDH Intersection:  
   - dsl: test_intersect_job_dsl.json  
   - runtime_config : test_intersect_job_ecdh_conf.json

8. ECDH Intersection with Preprocessing:  
   - dsl: test_intersect_job_dsl.json  
   - runtime_config : test_intersect_job_ecdh_w_preprocess_conf.json
  
9. RSA Intersection with Cache:  
   - dsl: test_intersect_job_cache_dsl.json  
   - runtime_config : test_intersect_job_rsa_cache_conf.json
    
10. DH Intersection with Cache:  
    - dsl: test_intersect_job_cache_dsl.json  
    - runtime_config : test_intersect_job_dh_cache_conf.json
   
11. ECDH Intersection with Cache:  
    - dsl: test_intersect_job_cache_dsl.json  
    - runtime_config : test_intersect_job_ecdh_cache_conf.json
  
12. RSA Intersection with Cache Loader:  
    - dsl: test_intersect_job_cache_loader_dsl.json  
    - runtime_config : test_intersect_job_rsa_cache_loader_conf.json
 
13. Estimated Intersect Cardinality:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_rsa_cardinality_conf.json
 
14. Exact Intersect Cardinality with ECDH:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_ecdh_exact_cardinality_conf.json

15. Exact Intersect Cardinality with DH:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_dh_exact_cardinality_conf.json

16. DH Intersection with Preprocessing:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_dh_w_preprocess_conf.json

17. RSA Intersection with Preprocessing:  
    - dsl: test_intersect_job_dsl.json  
    - runtime_config : test_intersect_job_rsa_w_preprocess_conf.json

18. ECDH Intersection with Cache Loader:  
    - dsl: test_intersect_job_cache_loader_dsl.json  
    - runtime_config : test_intersect_job_ecdh_cache_loader_conf.json
 
19. Exact Multi-host Intersect Cardinality with ECDH:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_ecdh_multi_exact_cardinality_conf.json

20. Exact Multi-host Intersect Cardinality with DH:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_dh_multi_exact_cardinality_conf.json

21. Exact Multi-host Intersect with ECDH:
    - dsl: test_intersect_job_dsl.json
    - runtime_config: "test_intersect_job_ecdh_multi_conf.json


Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}
