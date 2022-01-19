## Hetero SSHE Poisson Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different types of tasks.

#### Example Task

1. Train_task:
    dsl: hetero_poisson_dsl.json
    runtime_config : hetero_poisson_conf.json

2. Cross Validation Task:
    dsl: hetero_poisson_cv_dsl.json
    runtime_config: hetero_poisson_cv_conf.json

3. Poisson with validation:
    dsl: hetero_poisson_validate_dsl.json
    conf: hetero_poisson_validate_conf.json

4. Poisson with Warm start task:
    dsl: hetero_poisson_warm_start_dsl.json
    conf: hetero_poisson_warm_start_conf.json

5. Predict_task:
    dsl: hetero_poisson_predict_dsl.json
    runtime_config : hetero_poisson_predict_conf.json


Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](hetero_lr_normal_predict_conf.json)