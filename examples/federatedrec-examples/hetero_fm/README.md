## Hetero Factorization Machine Configuration Usage Guide.

This section introduces the dsl and conf for usage of uploading dataset and training model of HeteroFM.

#### Upload data

We have provided upload config for you can upload example data conveniently.

breast data set
    1. Guest Party Data: upload_data_guest.json
    2. Host Party Data: upload_data_host.json

    This data set can be applied for train task, train & validation task.

#### Training Task.

Train and evaluation task:
    dsl: test_hetero_fm_train_job_dsl.json
    runtime_config : test_hetero_fm_train_job_conf.json

Users can use following commands to running the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the training task.