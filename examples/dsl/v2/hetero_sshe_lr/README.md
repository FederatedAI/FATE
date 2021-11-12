## Hetero SSHE LR Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different types of tasks.

#### Example Task

1. Train_task:
    dsl: hetero_lr_normal_dsl.json
    runtime_config : hetero_lr_normal_conf.json

2. LR Compute Loss:
    dsl: hetero_lr_compute_loss_dsl.json
    runtime_config: hetero_lr_compute_loss_conf.json

3. Cross Validation Task(with fold history data output of predict score):
    dsl: hetero_lr_cv_dsl.json
    runtime_config: hetero_lr_cv_conf.json

4. One vs Rest(OVR) Task:
    dsl: hetero_lr_ovr_dsl.json
    conf: hetero_lr_ovr_conf.json

5. LR with validation:
    dsl: hetero_lr_with_validate_dsl.json
    conf: hetero_lr_with_validate_conf.json

6. LR with Warm start task:
    dsl: hetero_lr_warm_start_dsl.json
    conf: hetero_lr_warm_start_conf.json

7. LR with Encrypted Reveal in Host task:
    dsl: hetero_lr_encrypted_reveal_in_host_dsl.json
    conf: hetero_lr_encrypted_reveal_in_host_conf.json

8. LR L1 penalty task:
    dsl: hetero_lr_l1_dsl.json
    conf: hetero_lr_l1_conf.json

9. OVR LR with Encrypted Reveal in Host task:
    dsl: hetero_lr_ovr_encrypted_reveal_in_host_dsl.json
    conf: hetero_lr_ovr_encrypted_reveal_in_host_conf.json

10. LR OVR None-penalty task:
    dsl: hetero_lr_ovr_none_penalty_dsl.json
    conf: hetero_lr_ovr_none_penalty_conf.json

11. LR OVR L1 penalty task:
    dsl: hetero_lr_ovr_l1_dsl.json
    conf: hetero_lr_ovr_l1_conf.json

12. LR with Large Init Weight:
    dsl: hetero_lr_large_init_w_compute_loss_dsl.json
    conf: hetero_lr_large_init_w_compute_loss_conf.json

13. LR without intercept:
    dsl: hetero_lr_normal_not_fit_intercept_dsl.json
    conf: hetero_lr_normal_not_fit_intercept_conf.json

14. LR Compute Loss without reveal:
    dsl: hetero_lr_compute_loss_not_reveal_dsl.json
    conf: hetero_lr_compute_loss_not_reveal_conf.json

15. LR Normal Predict:
    dsl: hetero_lr_normal_predict_dsl.json
    conf: hetero_lr_normal_predict_conf.json

16. LR OVR Predict:
    dsl: hetero_lr_ovr_predict_dsl.json
    conf: hetero_lr_ovr_predict_conf.json

Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](hetero_lr_normal_predict_conf.json)