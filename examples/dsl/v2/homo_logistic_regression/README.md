## Homo Logistic Regression Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Example Task.

1. Train Task:
    dsl: homo_lr_train_dsl.json
    runtime_config : homo_lr_train_conf.json
   
2. Train, test and evaluation task:
    dsl: homo_lr_train_eval_dsl.json
    runtime_config: homo_lr_train_eval_conf.json
   
3. Cross Validation Task:
    dsl: homo_lr_cv_dsl.json
    runtime_config: homo_lr_cv_conf.json

4. Multi-host Task:
    dsl: homo_lr_multi_host_dsl.json
    conf: homo_lr_multi_host_conf.json

    Please note that we use a same data set for every host. This is just a demo showing how tow config multi-host task

5. predict Task:
    dsl: homo-lr-normal-predict-dsl.json
    conf: homo-lr-normal-predict-conf.json

6. single_eval:
    dsl: homo_lr_eval_dsl.json
    conf: homo_lr_eval_conf.json

7. Multi-Class Train Task:
   dsl: homo_lr_one_vs_all_dsl.json
   conf: homo_lr_one_vs_all_conf.json

8. Multi-Class Train With Paillier Task:
   dsl: homo_lr_one_vs_all_encrypted_host_dsl.json
   conf: homo_lr_one_vs_all_encrypted_host_conf.json

Users can use following commands to running the task.
    
    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use it to predict, you can use the obtained model to perform prediction. You need to add the corresponding model id and model version to the configuration [file](./hetero-lr-normal-predict-conf.json)