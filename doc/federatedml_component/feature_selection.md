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

Below lists available input models and their corresponding filter methods(as parameters in configuration):

| Isometric Model 	| Filter Method                                                                                                                                                                                                                                                                                                                                                             	|
|-----------------	|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| None            	| [manually](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-manually-left.py) <br> [percentage_value](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-percentage-value.py)                                                                                                                           	|
| Binning         	| [iv_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-local-only.py)                                                                                                                                                                                                                                                             	|
| Statistic       	| [statistic_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-multi-iso.py)                                                                                                                                                                                                                                                       	|
| Pearson         	| [correlation_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-pearson.py)(with 'iv' metric & binning model) <br> [vif_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-vif.py)                                                                                                        	|
| SBT             	| [hetero_sbt_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-multi-iso.py) <br> [hetero_fast_sbt_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-fast-sbt.py) <br> [homo_sbt_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-homo-sbt.py) 	|
| PSI             	| [psi_filter](../../examples/pipeline/hetero_feature_selection/pipeline-hetero-feature-selection-multi-iso.py)                                                                                                                                                                                                                                                             	|

1.  unique\_value: filter the columns if all values in this feature is
    the same

2.    - iv\_filter: Use iv as criterion to selection features. Support
        three mode: threshold value, top-k and top-percentile.
        
          - threshold value: Filter those columns whose iv is smaller
            than threshold. You can also set different threshold for
            each party.
          - top-k: Sort features from larger iv to smaller and take top
            k features in the sorted result.
          - top-percentile. Sort features from larger to smaller and
            take top percentile.
        
        Besides, multi-class iv filter is available if multi-class iv
        has been calculated in upstream component. There are three
        mechanisms to select features. Please remind that there exist as
        many ivs calculated as the number of labels since we use
        one-vs-rest for multi-class cases.
        
          - "min": take the minimum iv among all results.
          - "max": take the maximum ones
        
        \* "average": take the average among all results. After that, we
        get unique one iv for each column so that we can use the three
        mechanism mentioned above to select features.

3.  statistic\_filter: Use statistic values calculate from DataStatistic
    component. Support coefficient of variance, missing value,
    percentile value etc. You can pick the columns with higher statistic
    values or smaller values as you need.

4.  psi\_filter: Take PSI component as input isometric model. Then, use
    its psi value as criterion of selection.

5.  hetero\_sbt\_filter/homo\_sbt\_filter/hetero\_fast\_sbt\_filter:
    Take secureboost component as input isometric model. And use feature
    importance as criterion of selection.

6.  manually: Indicate features that need to be filtered.

7.  percentage\_value: Filter the columns that have a value that exceeds
    a certain percentage.

Besides, we support multi-host federated feature selection for iv
filters. Hosts encode feature names and send the feature ids that are
involved in feature selection. Guest use iv filters' logic to judge
whether a feature is left or not. Then guest sends result back to hosts.
Hosts decode feature ids back to feature names and obtain selection
results.

![Figure 4: Multi-Host Selection
Principle\</div\>](../images/multi_host_selection.png)


<!-- mkdocs
## Param

::: federatedml.param.feature_selection_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->
