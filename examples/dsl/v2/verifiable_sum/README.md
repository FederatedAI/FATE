## Verifiable Sum Configuration Usage Guide.

 This section introduces the dsl and conf for usage of different tasks.

 1. Verifiable Sum Task:

     dsl: test_verifiable_sum_dsl.json

     runtime_config : test_verifiable_sum_conf.json

 Users can use following commands to run the task.

     flow job submit -c ${runtime_config} -d ${dsl}