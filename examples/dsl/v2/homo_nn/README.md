## Homo Neural Network Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task

- keras backend

    1. single_layer:
    
        dsl : test_homo_dnn_single_layer_dsl.json
        
        runtime_config: test_homo_dnn_single_layer_conf.json
       
    2. multi_layer:
    
        dsl: test_homo_dnn_multi_layer_dsl.json
        
        runtime_config: test_homo_dnn_multi_layer_conf.json
       
    3. multi_label and multi-host:
    
        dsl: test_homo_dnn_multi_label_dsl.json
        
        runtime_config: test_homo_dnn_multi_label_conf.json
    
    4. predict:
    
        dsl: test_homo_dnn_multi_layer_predict_dsl.json
        
        runtime_config: test_homo_dnn_multi_layer_predict_conf.json
 

Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}
