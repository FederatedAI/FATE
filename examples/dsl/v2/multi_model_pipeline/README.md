## Multi Pipeline Configuration Usage Guide.

This section introduces the dsl and conf for usage of a multi-pipeline task.

#### Multi Pipeline Task.

1. Multi Pipeline Task:
    dsl: test_multi_pipeline_dsl.json
    runtime_config : test_multi_pipeline_conf.json

Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

Note: the intersection output is only ids of intersection, because of the parameter of "only_output_key" in runtime_config.  
