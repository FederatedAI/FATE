## Local Baseline Configuration Usage Guide.

This section introduces the dsl and conf for usage of different tasks.

1. Hetero Train Task:

    dsl: test_local_baseline_job_dsl.json

    runtime_config : test_local_baseline_job_conf.json
    
    data type: multi-label

2.  Homo Train Task:

    dsl: test_local_baseline_homo_job_dsl.json

    runtime_config : test_local_baseline_homo_job_conf.json
    
    data type: binary label

Users can use following commands to run the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

After having finished a successful training task, you can use FATE Board to check model output and evaluation results. 