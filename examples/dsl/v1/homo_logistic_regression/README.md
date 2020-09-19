## Homo Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Upload data

We have provided several upload config for you can upload example data conveniently.

1. breast data set
    1. Guest Party Data: upload_data_guest.json
    2. Host Party Data: upload_data_host.json

This is suitable for all tasks listed below

#### Training Task.

1. Train_task:
    dsl: test_homolr_train_job_dsl.json
    runtime_config : test_homolr_train_job_conf.json
   
2. Train, test and evaluation task:
    dsl: test_homolr_evaluate_job_dsl.json
    runtime_config: test_homolr_evaluate_job_conf.json
   
3. Cross Validation Task:
    dsl: test_homolr_cv_job_dsl.json
    runtime_config: test_homolr_cv_job_conf.json

4. Multi-host Task:
    dsl: test_homolr_train_job_dsl.json
    conf: test_multi_host_job_conf.json

    Please note that we use a same data set for every host. This is just a demo showing how tow config multi-host task

5. Spark backend Task:
    dsl: test_homolr_train_job_dsl.json
    conf: test_spark_backend_job_conf.json
    This task is available if you have deploy spark backend.
    
Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task, you can use it to predict too.