## Sample Weight Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Balanced Mode Task:

    dsl: sample_weight_job_dsl.json

    runtime_config : sample_weight_balanced_job_conf.json


2. Column Name Task:

    dsl: sample_weight_name_job_dsl.json

    runtime_config : sample_weight_name_job_conf.json

3. Class Dictionary with Feature Engineering Task:

    dsl: sample_weight_class_dict_feature_engineering_job_dsl.json

    runtime_config : sample_weight_class_dict_feature_engineering_job_conf.json

4. Multi-host Balanced Mode Task:

    dsl: sample_weight_job_dsl.json

    runtime_config : sample_weight_multi_host_job_conf.json

5. Sample Weight Transform Task:

    dsl: sample_weight_transform_job_dsl.json

    runtime_config : sample_weight_balanced_job_conf.json

Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check output. 