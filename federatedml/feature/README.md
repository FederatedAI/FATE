# Feature Binning

Feature binning or data binning is a data pre-processing technique. It can be use to reduce the effects of minor observation errors, calculate information values and so on.

In this version, we provide a widely used binning method called quantile binning. To achieve this approach, we have used a special data structure mentioned in this [paper](https://www.researchgate.net/profile/Michael_Greenwald/publication/2854033_Space-Efficient_Online_Computation_of_Quantile_Summaries/links/0f317533ee009cd3f3000000/Space-Efficient-Online-Computation-of-Quantile-Summaries.pdf). Feel free to check out the detail algorithm in the paper.

We are looking forward more binning methods and more methods will come out soon.

# Feature Selection

Feature selection is a process that select subset of features for use in model construction. Take good advantage of feature selection can improve the performance of a model.

In this version, we provides several filter methods for feature selection.

1. unique_value: filter the columns if all values in this feature is the same

2. iv_value_thres: Use information value to filter columns. Filter those columns whose iv is smaller than threshold.

3. iv_percentile: Use information value to filter columns. A float ratio threshold need to be provided. Pick floor(ratio * feature_num) features with higher iv. If multiple features around the threshold are same, all those columns will be keep.

4. coefficient_of_variation_value_thres: Use coefficient of variation to judge whether filtered or not.

5. outlier_cols: Filter columns whose certain percentile value is larger than a threshold.

Note: iv_value_thres and iv_percentile should not exist at the same times

More feature selection methods will be provided. Please make a discussion in issues if you have any needs.

# Sample

Fate v0.2 supports sample method. 
Sample module supports two sample mode, they are Random sample mode and StratifiedSampler sample mode.
* in random mode, "downsample" and "upsample" method is provided, users 
can set the sample parameter "fractions", which is the sample ratio of data.
* in stratified mode, "downsample" and "upsample" method is also provided, 
users can set the sample parameter "fractions" two, but it should be a list of tuples of (label_i, ratio),
which means that the sample ratios of different labels of data set.

# Feature scale
Feature scale is a process that scale each feature along column. Now it supports min-max scale and standard scale. 
1. min-max scale: this estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between min and max value of each feature.
2. standard scale: standardize features by removing the mean and scaling to unit variance

# Feature impute
Feature impute is a transformer for missing value imputation. The datasets may contain missing value such as blanks, NaN, None or Null, which is incompatible with some algorithm like logistic regression. To get the better effect, We can replace the missing value with mean value of each column, as well as minimum value, maximum value or any other value you want. You can also regard some values in dataset as outlier values, and replace them.

# OneHot Encoder
OneHot encoding is a process by which category variables are converted to binary values. The detailed info could be found in [OneHot wiki](https://en.wikipedia.org/wiki/One-hot)