## Column Expand Configuration Usage Guide.

#### Example Tasks

This section introduces the dsl and conf for different types of tasks.

1. Column Expand Task:

    dsl: test_column_expand_job_dsl.json
    
    runtime_config : test_column_expand_job_conf.json
    

2. Column Expand Task with Anonymous Header:

    dsl: test_column_expand_job_dsl.json
    
    runtime_config : test_column_expand_anonymous_job_conf.json


Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}
