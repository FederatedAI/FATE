### Federated Machine Learning
This module provides various federated machine learning algorithms for users. 

### Algorithm List

#### 1. [DataIO](./util/README.md)
This component is typically the first component of a modeling task. It will transform user-uploaded date into Instance object which can be used for the following components.

Corresponding module name: DataIO

Data Input: DTable, values are raw data.
Data Output: Transformed DTable, values are data instance define in federatedml/feature/instance.py


#### 2. [Intersect](./statistic/intersect/README.md)
Compute intersect data set of two parties without leakage of difference set information. Mainly used in hetero scenario task.

Corresponding module name: Intersection

Data Input: DTable
Data Output: DTable which keys are occurred in both parties.


#### 3. [Federated Sampling](./feature/README.md)
Federated Sampling data so that its distribution become balance in each party.This module support both federated and standalone version

Corresponding module name: FederatedSample

Data Input: DTable
Data Output: the sampled data, supports both random and stratified sampling.


#### 4. [Feature Scale](./feature/README.md)
Module for feature scaling and standardization.

Corresponding module name: FeatureScale

Data Input: DTable, whose values are instances.
Data Output: Transformed DTable.
Model Output: Transform factors like min/max, mean/std.


#### 5. [Hetero Feature Binning](./feature/README.md)
With binning input data, calculates each column's iv and woe and transform data according to the binned information.

Corresponding module name: HeteroFeatureBinning

Data Input: DTable with y in guest and without y in host.
Data Output: Transformed DTable.
Model Output: iv/woe, split points, event counts, non-event counts etc. of each column.


#### 6. [OneHot Encoder](./feature/README.md)
Transfer a column into one-hot format.

Corresponding module name: OneHotEncoder
Data Input: Input DTable.
Data Output: Transformed DTable with new headers.
Model Output: Original header and feature values to new header map.


#### 7. [Hetero Feature Selection](./feature/README.md)
Provide 5 types of filters. Each filters can select columns according to user config.

Corresponding module name: HeteroFeatureSelection
Data Input: Input DTable.
Model Input: If iv filters used, hetero_binning model is needed.
Data Output: Transformed DTable with new headers and filtered data instance.
Model Output: Whether left or not for each column.


#### 8. [Hetero-LR](./logistic_regression/README.md)
Build hetero logistic regression module through multiple parties.

Corresponding module name: HeteroLR
Data Input: Input DTable.
Model Output: Logistic Regression model.


#### 9. [Homo-LR](./logistic_regression/README.md)
Build homo logistic regression module through multiple parties.

Corresponding module name: HomoLR
Data Input: Input DTable.
Model Output: Logistic Regression model.


#### 10. [Hetero Secure Boosting](./tree/README.md)
Build hetero secure boosting model through multiple parties.

Corresponding module name: HeteroSecureBoost

Data Input: DTable, values are instances.
Model Output: SecureBoost Model, consists of model-meta and model-param


#### 11. [Evaluation](./evaluation/README.md)
Output the model evaluation metrics for user.

Corresponding module name: Evaluation



More available algorithms are coming soon.
