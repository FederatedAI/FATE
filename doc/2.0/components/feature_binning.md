# Hetero Feature Binning

Feature binning or data binning is a data pre-processing technique. It
can be used to reduce the effects of minor observation errors, calculate
information values and so on.

Currently, we provide quantile binning and bucket binning methods. To
achieve quantile binning approach, we have used a special data structure
mentioned in this
[paper](https://www.researchgate.net/profile/Michael_Greenwald/publication/2854033_Space-Efficient_Online_Computation_of_Quantile_Summaries/links/0f317533ee009cd3f3000000/Space-Efficient-Online-Computation-of-Quantile-Summaries.pdf).
Feel free to check out the detail algorithm in the paper.

As for calculating the federated iv and woe values, the following figure
can describe the principle properly.

![Figure 1 (Federated Feature Binning
Principle)](../images/binning_principle.png)

As the figure shows, B party which has the data labels encrypt its
labels with Addiction homomorphic encryption and then send to A. A
static each bin's label sum and send back. Then B can calculate woe and
iv base on the given information.

For multiple hosts, it is similar with one host case. Guest sends its
encrypted label information to all hosts, and each of the hosts
calculates and sends back the static info.

![Figure 2： Multi-Host Binning
Principle](../images/multiple_host_binning.png)

## Features

1. Support Quantile Binning based on quantile summary algorithm.
2. Support Bucket Binning.
3. Support calculating woe and iv values.
4. Support transforming data into bin indexes or woe value(guest only).
5. Support multiple-host binning.
6. Support asymmetric binning methods on Host & Guest sides.

Below lists supported features with links to examples:

| Cases                                | Scenario                                                                                                                                                                             	                                |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input Data with Categorical Features | [bucket binning](../../../examples/pipeline/hetero_feature_binning/test_feature_binning_bucket.py) <br> [quantile binning](../../../examples/pipeline/hetero_feature_binning/test_feature_binning_quantile.py)        |
| Output Data Transformed              | [bin index](../../../examples/pipeline/hetero_feature_binning/test_feature_binning_asymmetric.py) <br> [woe value(guest-only)](.../../../examples/pipeline/hetero_feature_binning/test_feature_binning_asymmetric.py) |
| Skip Metrics Calculation             | [multi_host](../../../examples/pipeline/hetero_feature_binning/test_feature_binning_multi_host.py)                                           	                                                                        |


