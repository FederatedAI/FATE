## Hetero SecureBoost Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Binary-Class: 
    dsl: test_secureboost_train_dsl.json
    runtime_config : test_secureboost_train_binary_conf.json
   
2. Multi-Class:
    dsl: test_secureboost_train_dsl.json
    runtime_config: test_secureboost_train_multi_conf.json
   
3. Regression:
    dsl: test_secureboost_train_dsl.json
    runtime_config: test_secureboost_train_regression_conf.json
    
#### Cross Validation Class

1. Binary-Class:
    dsl: test_secureboost_cross_validation_dsl.json 
    runtime_config: test_secureboost_cross_validation_binary_conf.json 
    
2. Multi-Class:
    dsl: test_secureboost_cross_validation_binary_conf.json  
    runtime_config: test_secureboost_cross_validation_multi_conf.json  
    
3. Regression:
    dsl: test_secureboost_cross_validation_dsl.json 
    runtime_config: test_secureboost_cross_validation_regression_conf.json
    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.