## Hetero Feature Binning Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

    "large_bin"
        "conf": "hetero_binning_large_bin_conf.json",
        "dsl": "hetero_binning_large_bin_dsl.json"
    
    "optimal_gini"
        "conf": "hetero_binning_optim_gini_conf.json",
        "dsl": "hetero_binning_optim_gini_dsl.json"
    
    "optimal_sparse_gini"
        "conf": "hetero_binning_sparse_optimal_gini_conf.json",
        "dsl": "hetero_binning_sparse_optimal_gini_dsl.json"
    
    "optimal_iv"
        "conf": "hetero_binning_optim_iv_conf.json",
        "dsl": "hetero_binning_optim_iv_dsl.json"
    
    "optimal_sparse_iv"
        "conf": "hetero_binning_sparse_optimal_iv_conf.json",
        "dsl": "hetero_binning_sparse_optimal_iv_dsl.json"
    
    "optimal_ks"
        "conf": "hetero_binning_optim_ks_conf.json",
        "dsl": "hetero_binning_optim_ks_dsl.json"
    
    "optimal_sparse_ks"
        "conf": "hetero_binning_sparse_optimal_ks_conf.json",
        "dsl": "hetero_binning_sparse_optimal_ks_dsl.json"
    
    "quantile_binning"
        "conf": "hetero_binning_quantile_binning_conf.json",
        "dsl": "hetero_binning_quantile_binning_dsl.json"
    
    "bucket_binning"
        "conf": "hetero_binning_bucket_binning_conf.json",
        "dsl": "hetero_binning_bucket_binning_dsl.json"
    
    "sparse_bucket_binning"
        "conf": "hetero_binning_sparse_bucket_binning_conf.json",
        "dsl": "hetero_binning_sparse_bucket_binning_dsl.json"
    
    "woe_binning"
        "conf": "hetero_binning_woe_binning_conf.json",
        "dsl": "hetero_binning_woe_binning_dsl.json"
    
    "category"
        "conf": "hetero_binning_category_binning_conf.json",
        "dsl": "hetero_binning_category_binning_dsl.json"
    
    "optimal_sparse_chi_square"
        "conf": "hetero_binning_sparse_optimal_chi_square_conf.json",
        "dsl": "hetero_binning_sparse_optimal_chi_square_dsl.json"
    
    "optimal_chi_square"
        "conf": "hetero_binning_optim_chi_square_conf.json",
        "dsl": "hetero_binning_optim_chi_square_dsl.json"
    
    "multi_host"
        "conf": "hetero_binning_multi_host_conf.json",
        "dsl": "hetero_binning_multi_host_dsl.json"
    
    "multi_host_optimal"
        "conf": "hetero_binning_multi_host_optimal_conf.json",
        "dsl": "hetero_binning_multi_host_optimal_dsl.json"
    
    "multi_host_sparse_optimal"
        "conf": "hetero_binning_multi_host_sparse_optimal_conf.json",
        "dsl": "hetero_binning_multi_host_sparse_optimal_dsl.json"
    
    "asymmetric"
        "conf": "hetero_binning_asymmetric_conf.json",
        "dsl": "hetero_binning_asymmetric_dsl.json"
    
    "skip_statistic"
        "conf": "hetero_binning_skip_statistic_conf.json",
        "dsl": "hetero_binning_skip_statistic_dsl.json"


Users can use following commands to running the task.
    
    flow job submit -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the fitting task, you can use it to transform too.