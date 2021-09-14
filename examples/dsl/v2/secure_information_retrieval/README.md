## Secure Information Retrieval Configuration Usage Guide.

 This section introduces the dsl and conf for SIR task.

1. Secure Information Retrieval Task to Retrieve Select Feature(s):

    dsl: test_secure_information_retrieval_dsl.json
    
    runtime_config : test_secure_information_retrieval_conf.json

 Users can use following commands to run the task.

     flow -f submit_job -c ${runtime_config} -d ${dsl}