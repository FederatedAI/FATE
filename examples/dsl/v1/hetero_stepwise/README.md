## Hetero Stepwise Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Logistic Regression Model:  
    example-data: (1) guest: breast_hetero_mini_guest.csv  
                  (2) host: breast_hetero_mini_host.csv  
    dsl: test_hetero_stepwise_lr_dsl.json  
    runtime_config: test_hetero_stepwise_lr_conf.json
     
2. Linear Regression Model:  
    example-data: (1) guest: motor_hetero_mini_guest.csv
                  (2) host: motor_hetero_mini_host.csv  
    dsl: test_hetero_stepwise_linr_dsl.json  
    runtime_config: test_hetero_stepwise_linr_conf.json
   
3. Poisson Regression:  
    example-data: (1) guest: dvisits_hetero_guest.csv
                  (2) host: dvisits_hetero_host.csv  
    dsl: test_hetero_stepwise_poisson_dsl.json  
    runtime_config: test_hetero_stepwise_poisson_conf.json
    
   
Users should upload the data described above with specified table name and namespace in the runtime_config, 
then use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}

Alternatively, use [run_test](../../../test) to run the whole testsuite, by which all required data will be automatically uploaded. 
