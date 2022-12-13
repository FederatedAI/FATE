## Homo Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Example Task.

1. Binary Train Task:
    dsl: homo_nn_train_binary_dsl.json
    runtime_config : homo_nn_train_binary_conf.json
   
2. Multi Train Task:
    dsl: homo_nn_train_multi_dsl.json
    runtime_config: homo_nn_train_multi_conf.json
   
3. Binary Task and Aggregate every N epoch:
    dsl: homo_nn_aggregate_n_epoch_dsl.json
    runtime_config: homo_nn_aggregate_n_epoch_conf.json

4. Regression Task:
    dsl: homo_nn_train_regression_dsl.json
    conf: homo_nn_train_regression_conf.json


Users can use following commands to running the task.
    
    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](./hetero-lr-normal-predict-conf.json)