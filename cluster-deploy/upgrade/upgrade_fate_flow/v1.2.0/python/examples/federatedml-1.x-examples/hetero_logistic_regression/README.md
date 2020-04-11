## Hetero Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Upload data

We have provided several upload config for you can upload example data conveniently.

1. breast data set
    1. Guest Party Data: upload_data_guest.json
    2. Host Party Data: upload_data_host.json

    This data set can be applied for train task, train & validation task, cv task and lr with feature engineering task that list below.

2. vehicle data set
    1. Guest Party Data: upload_vehicle_guest.json
    2. Host Party Data: upload_vehicle_host.json

    This data set can be applied for multi-class task.


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
    dsl: test_hetero_lr_train_job_dsl.json
    conf: test_hetero_lr_job_conf_one_vs_rest.json

    Note: Please remind that one vs rest task need different data set. We have already provided one called "vehicle scale"

5. LR with feature engineering task
    dsl: test_lr_with_feature_engineering_dsl.json
    conf: test_lr_with_feature_engineering_job_conf.json

6. Multi-host training task:
    dsl: test_hetero_lr_train_job_dsl.json
    conf: test_multi_host_job_conf.json

7. Spark backend Task:
    dsl: test_hetero_lr_train_job_dsl.json
    conf: test_spark_backend_job_conf.json
    This task is available if you have deploy spark backend.

Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.