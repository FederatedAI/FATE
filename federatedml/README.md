### Federated Machine Learning
This module provides various federated machine learning algorithms for users. 

### Algorithm List

#### 1. DataIO
This component is typically the first component of a modeling task. It will transform user-uploaded date into Instance object which can be used for the following components.

Corresponding module name: DataIO

Data Input: DTable, values are raw data.
Data Output: Transformed DTable, values are data instance define in federatedml/feature/instance.py

[More details](./util/README.md)

#### 2. Intersect
Compute intersect data set of two parties without leakage of difference set information. Mainly used in hetero scenario task.

Corresponding module name: Intersection

Data Input: DTable
Data Output: DTable which keys are occurred in both parties.

[More details](./statistic/intersect/README.md)

#### 3. Federated Sampling
Federated Sampling data so that its distribution become balance in each party.This module support both federated and standalone version

Corresponding module name: FederatedSample

Data Input: DTable
Data Output: the sampled data, supports both random and stratified sampling.

[More details](./feature/README.md)

#### 4. Feature Scale
Module for feature scaling and standardization.

Corresponding module name: FeatureScale

Data Input: DTable, whose values are instances.
Data Output: Transformed DTable.
Model Output: Transform factors like min/max, mean/std.

[More details](./feature/README.md)

#### 5. Hetero Feature Binning
With binning input data, calculates each column's iv and woe and transform data according to the binned information.

Corresponding module name: HeteroFeatureBinning

Data Input: DTable with y in guest and without y in host.
Data Output: Transformed DTable.
Model Output: iv/woe, split points, event counts, non-event counts etc. of each column.

[More details](./feature/README.md)

#### 6. OneHot Encoder
Transfer a column into one-hot format.

Corresponding module name: OneHotEncoder
Data Input: Input DTable.
Data Output: Transformed DTable with new headers.
Model Output: Original header and feature values to new header map.

[More details](./feature/README.md)

#### 7. Hetero Feature Selection
Provide 5 types of filters. Each filters can select columns according to user config.

Corresponding module name: HeteroFeatureSelection
Data Input: Input DTable.
Model Input: If iv filters used, hetero_binning model is needed.
Data Output: Transformed DTable with new headers and filtered data instance.
Model Output: Whether left or not for each column.

[More details](./feature/README.md)

#### 8. Hetero-LR
Build hetero logistic regression module through multiple parties.

Corresponding module name: HeteroLR
Data Input: Input DTable.
Model Output: Logistic Regression model.

[More details](./logistic_regression/README.md)

#### 9. Homo-LR
Build homo logistic regression module through multiple parties.

Corresponding module name: HomoLR
Data Input: Input DTable.
Model Output: Logistic Regression model.

[More details](./logistic_regression/README.md)

#### 10. Hetero Secure Boosting
Build hetero secure boosting model through multiple parties.

Corresponding module name: HeteroSecureBoost

Data Input: DTable, values are instances.
Model Output: SecureBoost Model, consists of model-meta and model-param

[More details](./tree/README.md)

#### 11. Evaluation
Output the model evaluation metrics for user.

Corresponding module name: Evaluation

[More details](./evaluation/README.md)


More available algorithms are coming soon.
