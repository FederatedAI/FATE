## Hetero Feature Binning Configuration Usage Guide.

This section introduces the dsl and conf for usage of different type of task.

#### Fit Task.

1. Fit quantile_binning task:
    dsl: test_hetero_binning_job_dsl.json
    runtime_config : test_hetero_binning_add_onehot_job_dsl.json

2. large_bin_nums_quantile task:
    "conf": "./test_hetero_binning_big_bin_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

3. multi_host_binning:
    "conf": "./multi_hosts_binning_job_conf.json",
    "dsl": "./test_hetero_binning_add_onehot_job_dsl.json"

4. woe_coding:
    "conf": "./binning_transform_woe_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

5. bucket_binning task:
    "conf": "./test_bucket_binning_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

6. sparse_bucket_binning task:
    "conf": "./test_sparse_bucket_binning_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

7. category_binning task:
    "conf": "./test_category_binning_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

8. optimal_iv task:
    "conf": "./test_optimal_binning_iv_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

9. optimal_gini task:
    "conf": "./test_optimal_binning_gini_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

10. optimal_chi_square task:
    "conf": "./test_optimal_binning_chi_square_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

11. optimal_ks task:
    "conf": "./test_optimal_binning_ks_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

12. sparse_optimal_iv task:
    "conf": "./test_sparse_optimal_binning_iv_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

13. sparse_optimal_gini task:
    "conf": "./test_sparse_optimal_binning_gini_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

14. sparse_optimal_chi_square task:
    "conf": "./test_sparse_optimal_binning_chi_square_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

15. sparse_optimal_ks task:
    "conf": "./test_sparse_optimal_binning_ks_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

16. multi_host_optimal task:
    "conf": "./multi_hosts_optimal_binning_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

17. multi_host_sparse_optimal task:
    "conf": "./multi_hosts_optimal_binning_job_conf.json",
    "dsl": "./test_hetero_binning_job_dsl.json"

Users can use following commands to running the task.
    
    python {fate_install_path}/fate_flow/fate_flow_client.py -f submit_job -c ${runtime_config} -d ${dsl}

Moreover, after successfully running the fitting task, you can use it to transform too.