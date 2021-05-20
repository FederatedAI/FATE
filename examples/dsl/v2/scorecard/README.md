## Scorecard Configuration Usage Guide.

This section introduces the dsl and conf for usage of different tasks.

1. Credit Scorecard Task:

    dsl: test_scorecard_job_dsl.json

    runtime_config : test_scorecard_job_conf.json

Users can use following commands to run the task.

    flow job submit -c ${runtime_config} -d ${dsl}