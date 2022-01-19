## Hetero SSHE Linear Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different types of tasks.

#### Example Task

1. Train_task:
    dsl: hetero_linr_dsl.json
    runtime_config : hetero_linr_conf.json

2. LinR Compute Loss without Reveal:
    dsl: hetero_linr_compute_loss_not_reveal_dsl.json
    runtime_config: hetero_linr_compute_loss_not_reveal_conf.json

3. Cross Validation Task:
    dsl: hetero_linr_cv_dsl.json
    runtime_config: hetero_linr_cv_conf.json

4. LinR with validation:
    dsl: hetero_linr_validate_dsl.json
    conf: hetero_linr_validate_conf.json

5. LinR with Warm start task:
    dsl: hetero_linr_warm_start_dsl.json
    conf: hetero_linr_warm_start_conf.json

6. LinR with Encrypted Reveal in Host task:
    dsl: hetero_linr_encrypted_reveal_in_host_dsl.json
    conf: hetero_linr_encrypted_reveal_in_host_conf.json

7. LinR with Large Init Weight:
    dsl: hetero_linr_large_init_w_compute_loss_dsl.json
    conf: hetero_linr_large_init_w_compute_loss_conf.json

8. LinR with sample weight:
    dsl: hetero_linr_sample_weight_dsl.json
    conf: hetero_linr_sample_weight_conf.json

9. Predict_task:
    dsl: hetero_linr_predict_dsl.json
    runtime_config : hetero_linr_predict_conf.json


Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](hetero_lr_normal_predict_conf.json)