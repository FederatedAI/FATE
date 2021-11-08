# Data Statistic

This component will do some statistical work on the data, including
statistical mean, maximum and minimum, median, etc.

The indicators for each column that can be statistic are list as follow.

1.  count: Number of data
2.  sum: The sum of this column
3.  mean: The mean of this column
4.  variance/stddev: Variance and standard deviation of this column
5.  median: Median of this column
6.  min/max: Min and Max value of this column
7.  coefficient of variance: The formula is abs(stddev / mean)
8.  missing\_count/missing\_ratio: Number and ratio of missing value in
    this column
9.  skewness: The definition can be referred to
    [here](https://en.wikipedia.org/wiki/Skewness)
10. kurtosis: The definition can be referred to
    [here](https://en.wikipedia.org/wiki/Kurtosis)
11. percentile: The value of percentile. Accept 0% to 100% while the
    number before the "%" should be integer.

These static values can be used in feature selection as a criterion.

<!-- mkdocs
## Param

::: federatedml.param.statistics_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

<!-- mkdocs
## Examples
{% include-examples "data_statistics" %}
-->
