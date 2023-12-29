# Hetero Feature Selection

Feature selection is a process that selects a subset of features for
model construction. Taking advantage of feature selection can improve
model performance.

In this version, we provide several filter methods for feature
selection. Note that module works in a cascade manner where
selected result of filter A will be input into next filter B.
User should pay attention to the order of listing when
supplying multiple filters to `filter_methods` param in job configuration.

## Features

Below lists available input models and their corresponding filter methods with links to examples:

| Input Models      | Filter Method                                                                                                                                                                                  	                                                                                                                                                            |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| None            	 | [manual](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_manual.py)                                                                                                                                             	                                                                                                                |
| Binning         	 | [iv_filter(threshold)](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_binning.py) <br> [iv_filter(top_k)](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_multi_model.py) <br> [iv_filter(top_percentile)](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_multi_host.py) |
| Statistic       	 | [statistic_filter](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_statistics.py)                                                                                                                                                                                                                                                |

Most of the filter methods above share the same set of configurable parameters.
Below lists their acceptable parameter values.

| Filter Method                     	 | Parameter Name  	 | metrics                                                                                                                                                | filter_type                            	 | take_high  	 |
|-------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------|
| IV Filter                         	 | filter_param    	 | "iv"                                                                                                                                                   | "threshold", "top_k", "top_percentile" 	 | True       	 |
| Statistic Filter                  	 | statistic_param 	 | "max", "min", "mean", "median", "std", "var", "coefficient_of_variance", "skewness", "kurtosis", "missing_count", "missing_ratio", quantile(e.g."95%") | "threshold", "top_k", "top_percentile" 	 | True/False 	 |

1.
    - iv\_filter: Use iv as criterion to selection features. Support
      three mode: threshold value, top-k and top-percentile.

        - threshold value: Filter those columns whose iv is smaller
          than threshold. You can also set different threshold for
          each party.
        - top-k: Sort features from larger iv to smaller and take top
          k features in the sorted result.
        - top-percentile. Sort features from larger to smaller and
          take top percentile.

2. statistic\_filter: Use statistic values calculate from DataStatistic
   component. Support coefficient of variance, missing value,
   percentile value etc. You can pick the columns with higher statistic
   values or smaller values as you need.

3. manually: Indicate features that need to be filtered or kept.

Besides, we support multi-host federated feature selection for iv
filters. Starting in ver 2.0.0-beta, all data sets will obtain anonymous header
during transformation from local file. Guest use iv filters' logic to judge
whether a feature is left or not. Then guest sends result filter back to hosts.
During this selection process, guest will not know the real name of host(s)' features.

![Figure 4: Multi-Host Selection
Principle\</div\>](../../images/multi_host_selection.png)