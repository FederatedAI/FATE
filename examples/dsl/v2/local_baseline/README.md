## Local Baseline Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Hetero Train Task:

    dsl: test_local_baseline_job_dsl.json

    runtime_config : test_local_baseline_job_conf.json
    
    data type: multi-class label

2. Hetero Predict Task:

    dsl: test_local_baseline_predict_job_dsl.json

    runtime_config : test_local_baseline_predict_job_conf.json
    
    data type: multi-class label

3.  Homo Train Task:

    dsl: test_local_baseline_homo_job_dsl.json

    runtime_config : test_local_baseline_homo_job_conf.json
    
    data type: binary label
    
4.  Homo Predict Task:

    dsl: test_local_baseline_homo_predict_job_dsl.json

    runtime_config : test_local_baseline_homo_predict_job_conf.json
    
    data type: binary label

5. Hetero Train Task with Sample Weight:

    dsl: test_local_baseline_job_dsl.json

    runtime_config : test_local_baseline_sample_weight_job_conf.json
    
    data type: multi-class label

Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check model output and evaluation results. 