## Hetero Feature Selection Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

    "iv-top-k": {
        "conf": "hetero_feature_selection_iv_top_k_conf.json",
        "dsl": "hetero_feature_selection_iv_top_k_dsl.json"
    },
    "iv-selection": {
        "conf": "hetero_feature_selection_iv_selection_conf.json",
        "dsl": "hetero_feature_selection_iv_selection_dsl.json"
    },
    "multi-host": {
        "conf": "hetero_feature_selection_multi_host_conf.json",
        "dsl": "hetero_feature_selection_multi_host_dsl.json"
    },
    "selection": {
        "conf": "hetero_feature_selection_selection_conf.json",
        "dsl": "hetero_feature_selection_selection_dsl.json"
    },
    "manually": {
        "conf": "hetero_feature_selection_manually_conf.json",
        "dsl": "hetero_feature_selection_manually_dsl.json"
    },
    "select-cols": {
        "conf": "hetero_feature_selection_select_cols_conf.json",
        "dsl": "hetero_feature_selection_select_cols_dsl.json"
    },
    "select-col-names": {
        "conf": "hetero_feature_selection_select_col_names_conf.json",
        "dsl": "hetero_feature_selection_select_col_names_dsl.json"
    },
    "percentage-value": {
        "conf": "hetero_feature_selection_percentage_value_conf.json",
        "dsl": "hetero_feature_selection_percentage_value_dsl.json"
    },
    "manually-left": {
        "conf": "hetero_feature_selection_manually_left_conf.json",
        "dsl": "hetero_feature_selection_manually_left_dsl.json"
    },
    "multi-iso": {
        "conf": "hetero_feature_selection_multi_iso_conf.json",
        "dsl": "hetero_feature_selection_multi_iso_dsl.json"
    },
    "fast-sbt": {
        "conf": "hetero_feature_selection_fast_sbt_conf.json",
        "dsl": "hetero_feature_selection_fast_sbt_dsl.json"
    },
    "single-predict": {
        "conf": "hetero_feature_selection_single_predict_conf.json",
        "dsl": "hetero_feature_selection_single_predict_dsl.json"
    },
    "hetero_feature_selection_select_anonymous_col_names": {
            "conf": "hetero_feature_selection_select_anonymous_col_names_conf.json",
            "dsl": "hetero_feature_selection_select_anonymous_col_names_dsl.json"
    },
    "hetero_feature_selection_manually_anonymous": {
            "conf": "hetero_feature_selection_manually_anonymous_conf.json",
            "dsl": "hetero_feature_selection_manually_anonymous_dsl.json"
    },
    "hetero_feature_selection_manually_left_anonymous": {
            "conf": "hetero_feature_selection_manually_left_anonymous_conf.json",
            "dsl": "hetero_feature_selection_manually_left_anonymous_dsl.json"
    }
    
Users can use following commands to running the task.
    
    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the fitting task, you can use it to transform too.

For selection using anonymous feature name cases, make sure to change party id value in host feature names accordingly.
Alternatively, try pipeline examples, where party id will be automatically replaced.
