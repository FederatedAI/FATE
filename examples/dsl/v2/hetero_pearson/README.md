## Hetero Pearson Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Training Task.


1. Base Cross Parties Task:

    dsl: test_hetero_pearson_default_dsl.json

    runtime_config : test_hetero_pearson_default_conf.json

2. Host Only Task:

    dsl: test_hetero_pearson_host_only_dsl.json

    runtime_config : test_hetero_pearson_host_only_conf.json
    
3. Sole Task:

    dsl: test_hetero_pearson_sole_dsl.json

    runtime_config : test_hetero_pearson_sole_conf.json
   
4. Use Mix Rand Task:

    dsl: test_hetero_pearson_mix_rand_dsl.json

    runtime_config : test_hetero_pearson_mix_rand_conf.json
    
 
   
Users can use following commands to run a task.

    flow job submit -c ${runtime_config} -d ${dsl}
