# Data Statistic

This component will do some statistical work on the data, including
statistical mean, maximum and minimum, median, etc.

The indicators for each column that can be statistic are list as follow.

1. count: Number of data
2. sum: The sum of this column
3. mean: The mean of this column
4. variance/stddev: Variance and standard deviation of this column
5. median: Median of this column
6. min/max: Min and Max value of this column
7. coefficient of variance: The formula is abs(stddev / mean)
8. missing\_count/missing\_ratio: Number and ratio of missing value in
   this column
9. skewness: definition may be found
   [here](https://en.wikipedia.org/wiki/Skewness)
10. kurtosis: definition may be found
    [here](https://en.wikipedia.org/wiki/Kurtosis)
11. percentile: The value of percentile. Accept 0% to 100% while the
    number before the "%" should be integer.

For examples of running statistics

These statistic results can be used in feature selection as a criterion,
as in this [example](../../../examples/pipeline/hetero_feature_selection/test_feature_selection_statistics.py).
