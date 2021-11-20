## Homo Neural Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task

- keras backend

    1. single_layer:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: keras_homo_dnn_single_layer.json
       
    2. multi_layer:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: keras_homo_dnn_multi_layer.json
       
    3. multi_label and multi-host:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: keras_homo_dnn_multi_label.json
    
    4. multi_layer and predict
        
        dsl: homo_nn_dsl.json
        
        runtime_config: keras_homo_dnn_multi_layer_predict.json
    

- pytorch backend

    1. single_layer:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: pytorch_homo_dnn_single_layer.json
       
    2. multi_layer:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: pytorch_homo_dnn_multi_layer.json
       
    3. multi_label and multi-host:
    
        dsl: homo_nn_dsl.json
        
        runtime_config: pytorch_homo_dnn_multi_label.json
    

Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}
