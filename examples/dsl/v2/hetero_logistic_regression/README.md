## Hetero Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Example Task.

1. Train_task:
    dsl: hetero_lr_normal_dsl.json
    runtime_config : hetero_lr_normal_conf.json

2. Train, test and evaluation task:
    dsl: hetero_lr_validate_dsl.json
    runtime_config: hetero_lr_validate_conf.json

3. Cross Validation Task(with fold history data output of predict score):
    dsl: hetero_lr_cv_dsl.json
    runtime_config: hetero_lr_cv_conf.json

4. One vs Rest Task:
    dsl: hetero_lr_one_vs_all_dsl.json
    conf: hetero_lr_one_vs_all_conf.json

5. LR with feature engineering task
    dsl: hetero_lr_feature_engineering_dsl.json
    conf: hetero_lr_feature_engineering_conf.json

6. Multi-host training task:
    dsl: hetero_lr_multi_host_dsl.json
    conf: hetero_lr_multi_host_conf.json

7. lr_sparse training task:
    "conf": hetero_lr_sparse_conf.json,
    "dsl": hetero_lr_sparse_dsl.json

8. lr_sparse_sqn task:
    "conf": "hetero_lr_sparse_sqn_conf.json",
    "dsl": "hetero_lr_sparse_sqn_dsl.json"

9. lr_ovr_cv task:
    "conf": "hetero_lr_ovr_cv_conf.json",
    "dsl": "hetero_lr_ovr_cv_dsl.json"

10. lr_sparse_cv task:
    "conf": "hetero_lr_sparse_cv_conf.json",
    "dsl": "hetero_lr_sparse_cv_dsl.json"

11. lr_ovr_sqn task:
    "conf": "hetero_lr_ovr_sqn_conf.json",
    "dsl": "hetero_lr_ovr_sqn_dsl.json"

12. lr_sqn task:
    "conf": "hetero_lr_sqn_conf.json",
    "dsl": "hetero_lr_sqn_dsl.json"

13. early_stop_lr task:
    "conf": "hetero_lr_early_stop_conf.json",
    "dsl": "hetero_lr_early_stop_dsl.json"

14. Test Task:
    dsl: hetero-lr-normal-predict-dsl.json
    conf: hetero-lr-normal-predict-conf.json
    deps: Train_task
    
15. Warm start task:
    dsl: hetero_lr_warm_start_dsl.json
    conf: hetero_lr_warm_start_conf.json

Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](hetero_lr_normal_predict_conf.json)