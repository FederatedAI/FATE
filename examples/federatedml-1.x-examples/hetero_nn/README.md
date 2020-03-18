## Hetero Neural Network Configuration Usage Guide.

This section introduces the dsl and conf relationships for usage.

#### Training Task.

1. Binary-Class:  
    example-data: (1) guest: breast_b.csv  (2) host: breast_a.csv  
    dsl: test_hetero_nn_dsl.json 
    runtime_config: test_hetero_nn_keras.json
 
2. Multi-Class:  
    example-data: (1) guest: vehicle_scale_b.csv
                  (2) host: vehicle_scale_a.csv  
    dsl: test_hetero_nn_dsl.json 
    runtime_config: test_hetero_nn_keras_multi_label.json

3. Binary-Class With Early Stop Using
    example-data: (1) guest: breast_b.csv  (2) host: breast_a.csv  
    dsl: test_hetero_nn_dsl_with_early_stop.json 
    runtime_config: test_hetero_nn_keras_with_early_stop.json
    
   
Users can modify 'test_build_from_keras.py' under this folder to modify structure of hetero_nn, it will dump keras model to json string and replace keywords in 'test_hetero_nn_keras_temperate.json', then generates a file called 'test_hetero_nn_keras.json'.
 
Note: users should upload the data described above with specified table name and namespace in the runtime_config, 
then use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config{ -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
