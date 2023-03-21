## Label Transform Configuration Example Usage Guide.

#### Example Tasks

This section introduces the dsl & conf files for different types of tasks.

1. Label Transform Task(default mapping, no difference):

    - dsl: label_transform_dsl.json  
    - runtime_config : label_transform_conf.json

2. Label Transform Task with Encoder(with given label mapping):

    - dsl: label_transform_encoder_dsl.json  
    - runtime_config : label_transform_encoder_conf.json

3. Label Transform Task with Encoder(without label list):

    - dsl: label_transform_encoder_dsl.json  
    - runtime_config : label_transform_encoder_without_label_list_conf.json

4. Label Transform Task with Partially-specified Encoder(without label list):

    - dsl: label_transform_encoder_dsl.json  
    - runtime_config : label_transform_encoder_partial_conf.json

Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}
