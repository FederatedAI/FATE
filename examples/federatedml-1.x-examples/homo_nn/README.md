## Homo Neural Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task.

1. single_layer:
    dsl: test_homo_nn_train_then_predict.dsl
    runtime_config : test_homo_dnn_single_layer.json
   
2. multi_layer:
    dsl: test_homo_nn_train_then_predict
    runtime_config: test_homo_dnn_multi_layer.json
   
3. multi_label:
    dsl: test_homo_nn_train_then_predict
    runtime_config: test_homo_dnn_multi_label.json

    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.