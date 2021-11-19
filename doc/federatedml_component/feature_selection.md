# Hetero Feature Selection

Feature selection is a process that selects a subset of features for
model construction. Take good advantage of feature selection can improve
model performance.

In this version, we provide several filter methods for feature
selection.

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

## Features

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

More feature selection methods will be provided. Please make suggestions
by submitting an issue.
