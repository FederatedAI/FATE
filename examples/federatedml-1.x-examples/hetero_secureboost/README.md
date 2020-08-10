## Hetero SecureBoost Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Binary-Class:  
    example-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv  
   dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_binary_conf.json
     
2. Multi-Class:  
    example-data: (1) guest: vehicle_scale_hetero_guest.csv
                  (2) host: vehicle_scal_a.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_multi_conf.json
   
3. Regression:  
    example-data: (1) guest: student_hetero_guest.csv
                  (2) host: student_hetero_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_regression_conf.json
    
4. Multi-Host Regression  
    example-data: (1) guest: motor_hetero_guest.csv
                  (2) host1: motor_hetero_host_1.csv; 
                  (3) host2: motor_hetero_host_2.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_regression_multi_host_conf.json
    
5. Binary-Class With Missing Value  
    example-data: (1) guest: ionosphere_scale_hetero_guest.csv
                  (2) host: ionosphere_scale_hetero_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_binary_with_missing_value_conf.json  
    
    This example also contains another two feature since FATE-1.1.  
    (1) evaluate data during training process, check the "validation_freqs" field in runtime_config  
    (2) another homomorphic encryption method "Iterative Affine", check "encrypt_param" field in runtime_config.
    
6. Early stopping example
    example-data: (1) guest: student_hetero_guest.csv
                  (2) host: student_hetero_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_with_early_stopping_conf.json
    

#### Cross Validation Class

1. Binary-Class:  
    example-data: (1) guest: breast_hetero_guest.csv
                  (2) host: breast_hetero_guest.csv  
    dsl: test_secureboost_cross_validation_dsl.json  
    runtime_config: test_secureboost_cross_validation_binary_conf.json 
    
2. Multi-Class:  
    example-data: (1) guest: vehicle_scale_hetero_guest.csv
                  (2) host: vehicle_scal_a.csv  
    dsl: test_secureboost_cross_validation_binary_conf.json  
    runtime_config: test_secureboost_cross_validation_multi_conf.json  
    
3. Regression:  
    example-data: (1) guest: student_hetero_guest.csv
                  (2) host: student_hetero_host.csv  
    dsl: test_secureboost_cross_validation_dsl.json  
    runtime_config: test_secureboost_cross_validation_regression_conf.json
    
Users should upload the data described above with specified table name and namespace in the runtime_config, 
then use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
