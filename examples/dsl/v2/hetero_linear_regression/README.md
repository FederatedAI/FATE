## Hetero Linear Regression Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Train Task:

    dsl: test_hetero_linr_train_job_dsl.json

    runtime_config : test_hetero_linr_train_job_conf.json

2. Predict Task:
    
    dsl: test_hetero_linr_predict_job_dsl.json

    runtime_config: test_hetero_linr_predict_job_conf.json

3. Validate Task (with early-stopping parameters specified):

    dsl: test_hetero_linr_validate_job_dsl.json

    runtime_config : test_hetero_linr_validate_job_conf.json

4. Cross Validation Task:

    dsl: test_hetero_linr_cv_job_dsl.json

    runtime_config: test_hetero_linr_cv_job_conf.json

5. Multi-host Train Task:

    dsl: test_hetero_linr_train_job_dsl.json

    conf: test_hetero_linr_multi_host_train_job_conf.json

6. Multi-host Predict Task:

    dsl: test_hetero_linr_multi_host_predict_job_dsl.json

    conf: test_hetero_linr_multi_host_predict_job_conf.json

7. Multi-host Cross Validation Task:

    dsl: test_hetero_linr_multi_host_cv_job_dsl.json

    conf: test_hetero_linr_multi_host_cv_job_conf.json

8. Train Task with Sparse Data:
    
     dsl: test_hetero_linr_train_sparse_job_job_dsl.json

    runtime_config : test_hetero_linr_train_sparse_job_conf.json


Users can use following commands to run a task.

    bash flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](./test_hetero_linr_predict_job_conf.json)