## Feature Scale Configuration Usage Guide.

This section introduces the dsl and conf for running feature scale tasks.

#### Multi Pipeline Task.

1. Feature Scale Normal Mode Task(with `feat_upper` given in list):
    conf: test_feature_scale_normal_conf.json
    dsl: test_feature_scale_normal_dsl.json


2. Feature Scale Cap Mode Task(with integer `feat_upper` & `feat_lower`):
    conf: test_feature_scale_cap_conf.json
    dsl: test_feature_scale_cap_dsl.json

Users can use following commands to running the task.

    flow job submit -c ${runtime_config} -d ${dsl}

Note: the intersection output is only ids of intersection, because of the parameter of "only_output_key" in runtime_config.  
