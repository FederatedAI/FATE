## Scorecard Configuration Usage Guide.

This section introduces the dsl and conf for usage of different tasks.

1. Credit Scorecard Task:

    dsl: test_scorecard_job_dsl.json

    runtime_config : test_scorecard_job_conf.json

Users can use following commands to run the task.

    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}