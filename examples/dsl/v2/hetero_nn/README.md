## Hetero Neural Network Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Binary Train Task:

    example-data: (1) guest: breast_hetero_guest.csv (2) host: breast_hetero_host.csv  
    
    dsl: test_hetero_nn_dsl.json

    runtime_config : test_hetero_nn_binary_conf.json

2. Multi-label Train Task:
    
    example-data: (1) guest: vehicle_scale_hetero_guest.csv  
                  (2) host: vehicle_scale_hetero_host.csv  
    
    dsl: test_hetero_nn_dsl.json

    runtime_config: test_hetero_nn_multi_conf.json

3. Train Task With Early Stopping Strategy:

    example-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv  
    
    dsl: test_hetero_nn_binary_with_early_stop_dsl.json

    runtime_config : test_hetero_nn_binary_with_early_stop_conf.json

    This feature support since FATE-1.4, please have a look at the param "early_stopping_rounds", "metric", "validation_freqs"

4. Train Task With Selective BackPropagation Strategy:

    example-data: (1) guest: default_credit_hetero_guest.csv  (2) host: default_credit_hetero_host.csv  
    
    dsl: test_hetero_nn_dsl.json

    runtime_config : test_hetero_nn_binary_selective_bp_conf.json

    This feature support since FATE-1.6, please have a look at the param "selector_param"

5. Train Task With Interactive Layer DropOut Strategy:

    example-data: (1) guest: default_credit_hetero_guest.csv  (2) host: default_credit_hetero_host.csv  
    
    dsl: test_hetero_nn_dsl.json

    runtime_config : test_hetero_nn_binary_drop_out_conf.json

    This feature support since FATE-1.6, please have a look at the param "drop_out_keep_rate"

6. Train Task With Floating Point Precision Optimization:

    example-data: (1) guest: default_credit_hetero_guest.csv  (2) host: default_credit_hetero_host.csv  
    
    dsl: test_hetero_nn_dsl.json

    runtime_config : test_hetero_nn_binary_floating_point_precision_conf.json

    This feature support since FATE-1.6, please have a look at the param "floating_point_precision"

7. Train Task With Warm Start:  

    examples-data: (1) guest: breast_hetero_guest.csv  (2) host: breast_hetero_host.csv    
    
    dsl: test_hetero_nn_binary_with_warm_start_dsl.json  
    
    runtime_conf: test_hetero_nn_binary_with_warm_start_conf.json  
    
8. Train Task With CheckPoint:  
 
    script: pipeline-hetero-nn-train-with-check-point.py  
    
    dsl: test_hetero_nn_binary_with_check_point_dsl.json    
    
    runtime_conf: test_hetero_nn_binary_with_check_point_conf.json  
    
Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.
