## Secret Sharing Sum Configuration Usage Guide.

 This section introduces the dsl and conf for usage of different tasks.

 1. Secret Sharing Sum Task:

     dsl: test_secret_sharing_sum_dsl.json

     runtime_config : test_secret_sharing_sum_conf.json

 Users can use following commands to run the task.

     python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}