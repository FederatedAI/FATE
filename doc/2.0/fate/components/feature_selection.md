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
| None            	 | [manual](../../../../examples/pipeline/hetero_feature_selection/test_feature_selection_manual.py)                                                                                                                                             	                                                                                                                |
| Binning         	 | [iv_filter(threshold)](../../../../examples/pipeline/hetero_feature_selection/test_feature_selection_binning.py) <br> [iv_filter(top_k)](../../../../examples/pipeline/hetero_feature_selection/test_feature_selection_multi_model.py) <br> [iv_filter(top_percentile)](../../../../examples/pipeline/hetero_feature_selection/test_feature_selection_multi_host.py) |
| Statistic       	 | [statistic_filter](../../../../examples/pipeline/hetero_feature_selection/test_feature_selection_statistics.py)                                                                                                                                                                                                                                                |

Most of the filter methods above share the same set of configurable parameters.
Below lists their acceptable parameter values.

| Filter Method                     	 | Parameter Name  	 | metrics                                                                                                                                                | filter_type                            	 | take_high  	 |
|-------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------|
| IV Filter                         	 | filter_param    	 | "iv"                                                                                                                                                   | "threshold", "top_k", "top_percentile" 	 | True       	 |
| Statistic Filter                  	 | statistic_param 	 | "max", "min", "mean", "median", "std", "var", "coefficient_of_variance", "skewness", "kurtosis", "missing_count", "missing_ratio", quantile(e.g."95%") | "threshold", "top_k", "top_percentile" 	 | True/False 	 |

## Filter Configuration

1. iv\_filter: Use iv as criterion to selection features. 
    - filter_type: Support three modes: threshold value, top-k and top-percentile.
        - threshold value: Filter those columns whose iv is smaller
          than threshold. You can also set different threshold for
          each party.
        - top-k: Sort features from larger iv to smaller and take top
          k features in the sorted result.
        - top-percentile. Sort features from larger to smaller and
          take top percentile.
    - select_federated: If set to True, the feature selection will be
      performed in a federated manner. The feature selection will be
      performed on the guest side, and the anonymously selected features will be
      sent to the host side. The host side will then filter the
      features based on the selected features from the guest side. This param is available in iv\_filter only.
    - threshold: The threshold value for feature selection.
    - take_high: If set to True, the filter will select features with
      higher iv values. If set to False, the filter will select
      features with lower iv values.
    - host_filter_type: The filter type for host features. It can be
      "threshold", "top_k", "top_percentile". This param is available in iv\_filter only.
    - host_threshold: The threshold value for feature selection on host
      features. This param is available in iv\_filter only.
    - host_top_k: The top k value for feature selection on host features.
      This param is available in iv\_filter only.
2. statistic\_filter: Use statistic values calculate from DataStatistic
   component. Support coefficient of variance, missing value,
   percentile value etc. You can pick the columns with higher statistic
   values or smaller values as you need.
    - filter_type: Support three modes: threshold value, top-k and top-percentile.
      - threshold value: Filter those columns whose statistic metric is smaller
        than threshold. You can also set different threshold for
        each party.
      - top-k: Sort features from larger statistic metric  to smaller and take top
        k features in the sorted result.
      - top-percentile. Sort features from larger to smaller and
        take top percentile.
    - threshold: The threshold value for feature selection.
    - take_high: If set to True, the filter will select features with
      higher metric values. If set to False, the filter will select
      features with lower iv values.

3. manually: Indicate features that need to be filtered or kept.
    - keep_col: The columns that need to be kept.
    - filter_out_col: The columns that need to be dropped.

Besides, we support multi-host federated feature selection for iv
filters. Starting in ver 2.0.0-beta, all data sets will obtain anonymous header
during transformation from local file. Guest use iv filters' logic to judge
whether a feature is left or not. Then guest sends result filter back to hosts.
During this selection process, guest will not know the real name of host(s)' features.

![Figure 4: Multi-Host Selection
Principle\</div\>](../../images/multi_host_selection.png)