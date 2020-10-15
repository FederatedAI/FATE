## Hetero FTL Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Plain Mode:  

    example-data: 
    
    (1) guest: nus_wide_train_guest.csv;nus_wide_validate_guest.csv 
    
    (2) host: nus_wide_train_host.csv;nus_wide_validate_host.csv 
     
    dsl: test_ftl_dsl.json
    
    runtime_config: test_ftl_conf.json
     
2. Predict:  

    example-data: 
    
    (1) guest: nus_wide_train_guest.csv;nus_wide_validate_guest.csv 
    
    (2) host: nus_wide_train_host.csv;nus_wide_validate_host.csv 
    
    dsl: test_ftl_dsl.json
    
    runtime_config: test_ftl_predict.json
   
3. Encrypted Mode: 
 
    example-data: 
    
    (1) guest: nus_wide_train_guest.csv;nus_wide_validate_guest.csv 
    
    (2) host: nus_wide_train_host.csv;nus_wide_validate_host.csv 
    
    dsl: test_ftl_dsl.json
    
    runtime_config: test_ftl_encrypted_conf.json
    
4. Communication-Efficient:   

    example-data: 
    
    (1) guest: nus_wide_train_guest.csv;nus_wide_validate_guest.csv 
    
    (2) host: nus_wide_train_host.csv;nus_wide_validate_host.csv 
    
    dsl: test_ftl_dsl.json
    
    runtime_config: test_ftl_comm_eff_conf.json


Note: users should upload the data described above with specified table name and namespace in the runtime_config, 
then use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.