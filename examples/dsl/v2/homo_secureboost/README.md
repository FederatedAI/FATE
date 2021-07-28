## Homo SecureBoost Configuration Usage Guide.

#### Example Tasks.

1. Binary-Class:  

    example-data: 
    
    (1) guest: breast_homo_guest.csv  (2) host: breast_homo_host.csv (3) test: breast_homo_test.csv 
     
    dsl: test_secureboost_train_dsl.json  
    
    runtime_config: test_secureboost_train_binary_conf.json
     
2. Multi-Class:  

    example-data: 
    
    (1) guest: vehicle_scale_homo_guest.csv (2) host: vehicle_scale_homo_host.csv (3) test: vehicle_scale_homo_test.csv  
             
    dsl: test_secureboost_train_dsl.json  
    
    runtime_config: test_secureboost_train_multi_conf.json
   
3. Regression: 
 
    example-data: 
    
    (1) guest: student_homo_guest.csv (2) host: student_homo_host.csv  (3) test: student_homo_test.csv 
           
    dsl: test_secureboost_train_dsl.json  
    
    runtime_config: test_secureboost_train_regression_conf.json
    
4. Multi-Host Binary-Class:   

    example-data: 
    
    (1) guest:  default_credit_homo_guest.csv  (2) host1:  default_credit_homo_host_1.csv  (3) host2:  default_credit_homo_host_2.csv  (4) test:  default_credit_homo_test.csv  
                
    dsl: test_secureboost_train_dsl.json  
    
    runtime_config: test_secureboost_train_binary_multi_host_conf.json
    
5. Binary-Class With Missing Value:  

    example-data: (1) guest: ionosphere_scale_hetero_guest.csv (2) host: ionosphere_scale_hetero_guest.csv
                   
    dsl: test_secureboost_train_dsl.json  
    
    runtime_config: test_secureboost_train_binary_with_missing_value_conf.json  
    
6. Predict:
    
    (1) test: breast_homo_test.csv
       
    dsl: test_predict_dsl.json
    
    runtime_config: test_predict_conf.json
    
#### Cross Validation Class

1. Binary-Class:  

    example-data: (1) guest: breast_homo_guest.csv  (2) host: breast_homo_host.csv  
    
    dsl: test_secureboost_cross_validation_dsl.json  
    
    runtime_config: test_secureboost_cross_validation_binary_conf.json 
    
2. Multi-Class:  

    example-data: (1) guest: vehicle_scale_homo_guest.csv
                  (2) host: vehicle_scale_homo_host.csv  
                  
    dsl: test_secureboost_cross_validation_dsl.json
    
    runtime_config: test_secureboost_cross_validation_multi_conf.json  
    
3. Regression:  

    example-data: (1) guest: student_homo_guest.csv
                  (2) host: student_homo_host.csv 
                  
    dsl: test_secureboost_cross_validation_dsl.json  
    
    runtime_config: test_secureboost_cross_validation_regression_conf.json
    

Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
