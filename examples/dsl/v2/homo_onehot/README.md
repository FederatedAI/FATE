## Homo OneHot Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Example Task.

1. Train_task:
    dsl: homo_onehot_test_dsl.json
    runtime_config : homo_onehot_test_conf.json
   
Users can use following commands to running the task.
    
    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](./hetero-lr-normal-predict-conf.json)