## Hetero Feature Binning Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

1. Fit task:
    dsl: test_hetero_binning_job_dsl.json
    runtime_config : test_hetero_binning_job_conf.json

2. Multi-host task:
    dsl: test_hetero_binning_job_dsl.json
    runtime_config : multi_hosts_binning_job_conf.json

3. Transform woe task:
    dsl: test_hetero_binning_job_dsl.json
    runtime_config : binning_transform_woe_job_conf.json

Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the fitting task, you can use it to transform too.