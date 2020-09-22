## Hetero Poisson Regression Configuration Usage Guide.

#### Upload data

We have provided several upload config for you can upload example data conveniently.

1. dvisits set
    1. Guest Party Data: test_hetero_poisson_upload_data_guest.json
    2. Host Party Data: test_hetero_poisson_upload_data_host.json

    This data set can be applied for train task, train & validation task, cv task and lr with feature engineering task that list below.

#### Example Tasks

This section introduces the dsl and conf for usage of different tasks.

1. Train Task:

    dsl: test_hetero_poisson_train_job_dsl.json

    runtime_config : test_hetero_poisson_train_job_conf.json
    (with exposure variable column name specified)

2. Predict Task:

    runtime_config: test_predict_conf.json
    
3.  Validate Task (with early-stopping parameter specified):

    dsl: test_hetero_poisson_validate_job_dsl.json

    runtime_config : test_hetero_poisson_validate_job_conf.json
  
4. Train Task with Sparse Data:
    
    dsl: test_hetero_poisson_train_job_dsl.json

    runtime_config : test_hetero_poisson_train_sparse_job_conf.json
    (with exposure variable column name specified)

5. Cross Validation Task:

    dsl: test_hetero_poisson_cv_job_dsl.json

    runtime_config: test_hetero_poisson_cv_job_conf.json

Users can use following commands to run the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](test_predict_conf.json).