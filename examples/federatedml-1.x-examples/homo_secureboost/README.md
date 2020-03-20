## Hetero SecureBoost Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Binary-Class:  
    example-data: (1) guest: homo_credit_guest.csv  (2) host: homo_credit_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_binary_conf.json
     
2. Multi-Class:  
    example-data: (1) guest: homo_vehicle_scale_guest.csv
                  (2) host: homo_vehicle_scale_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_multi_conf.json
   
3. Regression:  
    example-data: (1) guest: homo_student_guest.csv
                  (2) host: homo_student_host.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_regression_conf.json
    
4. Multi-Host Binary-Class   
    example-data: (1) guest:  credit_guest_multi.csv  
                  (2) host1:  credit_host1_multi.csv  
                  (3) host2:  credit_host2_multi.csv  
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_binary_multi_host_conf.json
    
5. Binary-Class With Missing Value  
    example-data: (1) guest: ionosphere_scale_b.csv
                  (2) host: ionosphere_scale_b.csv
    dsl: test_secureboost_train_dsl.json  
    runtime_config: test_secureboost_train_binary_with_missing_value_conf.json  
    
#### Cross Validation Class

1. Binary-Class:  
    example-data: (1) guest: homo_credit_guest.csv  (2) host: homo_credit_host.csv  
    dsl: test_secureboost_cross_validation_dsl.json  
    runtime_config: test_secureboost_cross_validation_binary_conf.json 
    
2. Multi-Class:  
    example-data: (1) guest: homo_vehicle_scale_guest.csv
                  (2) host: homo_vehicle_scale_host.csv  
    dsl: test_secureboost_cross_validation_binary_conf.json  
    runtime_config: test_secureboost_cross_validation_multi_conf.json  
    
3. Regression:  
    example-data: (1) guest: homo_student_guest.csv
                  (2) host: homo_student_host.csv 
    dsl: test_secureboost_cross_validation_dsl.json  
    runtime_config: test_secureboost_cross_validation_regression_conf.json