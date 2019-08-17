## Hetero Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task.

1. Train_task:
    dsl: test_hetero_lr_train_job_dsl.json
    runtime_config : test_hetero_lr_train_job_conf.json

2. Train, test and evaluation task:
    dsl: test_hetero_lr_validate_job_dsl.json
    runtime_config: test_hetero_lr_validate_job_conf.json

3. Cross Validation Task:
    dsl: test_hetero_lr_cv_job_dsl.json
    runtime_config: test_hetero_lr_cv_job_conf.json

4. One vs Rest Task:
    dsl: test_hetero_lr_job_dsl_one_vs_rest.json
    conf: test_hetero_lr_job_conf_one_vs_rest.json

    Note: Please remind that one vs rest task need different data set. We have already provided one called "vehicle scale"


Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.