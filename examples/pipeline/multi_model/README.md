## Multi Pipeline Configuration Usage Guide.

This section introduces the pipeline script of a multi-pipeline task.

#### Multi Pipeline Task.

1. Multi Pipeline Task:
    script: pipeline-multi-model.py

Users can use following commands to running the task.

    python ${pipeline_script}

Note: the intersection output is only ids of intersection, because of the parameter of "only_output_key" in runtime_config.  
