## Sample Weight Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Task:

    dsl: sample_weight_job_dsl.json

    runtime_config : sample_weightjob_conf.json


Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check output. 