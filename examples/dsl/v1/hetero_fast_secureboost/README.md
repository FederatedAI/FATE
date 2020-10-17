## Hetero SecureBoost Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Mix-Mode Binary-Class:  

    example-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv  
    
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_mix_binary_conf.json
     
2. Mix-Mode Multi-Class:  

    example-data: (1) guest: vehicle_scale_hetero_guest.csv
                  (2) host: vehicle_scale_hetero_host.csv
                  
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_mix_multiclass_conf.json
   
3. Mix-Mode Regression:  

    example-data: (1) guest: student_hetero_guest.csv
                  (2) host: student_hetero_host.csv  
                  
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_mix_regression_conf.json
    
4. Mix-Mode Multi-Host Binary: 

    example-data: (1) guest: breast_hetero_guest.csv
                  (2) host1: breast_hetero_host.csv
                  (3) host2: breast_hetero_host.csv 
                  
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_mix_multi_host_conf.json
    
5. Mix-Mode Predict:  

    example-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv  

    runtime_config: test_predict_conf.json
    
    
6. Layered-Mode Binary-Class:  

    example-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv  
    
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_layered_binary_conf.json
     
7. Layered-Mode Multi-Class:  

    example-data: (1) guest: vehicle_scale_hetero_guest.csv
                  (2) host: vehicle_scale_hetero_host.csv
                  
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_layered_multiclass_conf.json
   
8. Layered-Mode Regression:  

    example-data: (1) guest: student_hetero_guest.csv
                  (2) host: student_hetero_host.csv  
                  
    dsl: test_fast_sbt_dsl.json  
    
    runtime_config: test_fast_sbt_layered_regression_conf.json

    
Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
