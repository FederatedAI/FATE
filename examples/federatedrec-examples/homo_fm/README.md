## Homo Factorization Machine Configuration Usage Guide.

This section introduces the dsl and conf for usage of uploading dataset and training model of HomoFM.

#### Upload data

We have provided upload config for you can upload example data conveniently.

breast data set
    1. Guest Party Data: upload_data_guest.json
    2. Host Party Data: upload_data_host.json

    This data set can be applied for train task, train & validation task.

Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f upload -c upload_data_guest.json
    python {fate_install_path}/fate_flow/fate_flow_client.py -f upload -c upload_data_host.json


#### Training Task.

Train and evaluation task:
    dsl: test_homo_fm_train_job_dsl.json
    runtime_config : test_homo_fm_train_job_conf.json

Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task.