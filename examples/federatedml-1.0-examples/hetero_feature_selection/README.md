## Hetero Feature Selection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

1. Fit task:
    dsl: test_hetero_feature_selection_job_dsl.json
    runtime_config : test_hetero_feature_selection_job_conf.json

    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.