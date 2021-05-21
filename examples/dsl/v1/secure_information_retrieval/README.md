## Secure Information Retrieval Configuration Usage Guide.

 This section introduces the dsl and conf for usage of different tasks.

 1. Secure Information Retrieval Task:

     dsl: test_secure_information_retrieval_dsl.json

     runtime_config : test_secure_information_retrieval_conf.json

 Users can use following commands to run the task.

     python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}