## Hetero Feature Selection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

1. Fit task:
    dsl: test_hetero_feature_selection_job_dsl.json
    runtime_config : test_hetero_feature_selection_job_conf.json

2. iv_selection task:
    "conf": "test_iv_selection_job_conf.json",
    "dsl": "test_hetero_feature_selection_job_dsl.json"

3. Multi-host task:
    dsl: test_hetero_feature_selection_job_dsl.json
    runtime_config : test_hetero_feature_selection_multi_host_job_conf.json

4. selection_manually task:
    "conf": "./test_hetero_feature_selection_manually_job_conf.json",
    "dsl": "./test_hetero_feature_selection_job_dsl.json"

5. select_cols task:
    "conf": "./test_hetero_feature_selection_select_cols_job_conf.json",
    "dsl": "./test_hetero_feature_selection_job_dsl.json"

6. select_col_names task:
    "conf": "./test_hetero_feature_selection_select_cols_names_job_conf.json",
    "dsl": "./test_hetero_feature_selection_job_dsl.json"

7. single_predict task:
    "conf": "test_hetero_feature_eval_selection_job_conf.json",
    "dsl": "test_single_prediction_dsl.json"

Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the fitting task, you can use it to transform too.