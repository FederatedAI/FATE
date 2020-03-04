
English | [中文](./README_zh.md)
# Federated Machine Learning

FederatedML includes implementation of many common machine learning algorithms on federated learning. All modules are developed in a decoupling modular approach to enhance scalability. Specifically, we provide:

1. Federated Statistic: PSI, Union, Pearson Correlation, etc.

2. Federated Feature Engineering: Feature Sampling, Feature Binning, Feature Selection, etc.

3. Federated Machine Learning Algorithms: LR, GBDT, DNN, TransferLearning, which support Heterogeneous and Homogeneous styles.

4. Model Evaluation: Binary|Multiclass|Regression Evaluation, Local vs Federated Comparison.

5. Secure Protocol: Provides multiple security protocols for secure multi-party computing and interaction between participants.

 <div style="text-align:center", align=center>
<img src="../doc/images/federatedml_structure.png" alt="federatedml structure"/><br/>
Figure 1： Federated Machine Learning Framework
</div>

### Algorithm List

#### 1. [DataIO](./util/README.md)
This component is typically the first component of a modeling task. It will transform user-uploaded date into Instance object which can be used for the following components.

- Corresponding module name: DataIO

- Data Input: DTable, values are raw data.
- Data Output: Transformed DTable, values are data instance define in federatedml/feature/instance.py


#### 2. [Intersect](./statistic/intersect/README.md)
Compute intersect data set of two parties without leakage of difference set information. Mainly used in hetero scenario task.

- Corresponding module name: Intersection

- Data Input: DTable
- Data Output: DTable which keys are occurred in both parties.


#### 3. [Federated Sampling](./feature/README.md)
Federated Sampling data so that its distribution become balance in each party.This module support both federated and standalone version

- Corresponding module name: FederatedSample

- Data Input: DTable
- Data Output: the sampled data, supports both random and stratified sampling.


#### 4. [Feature Scale](./feature/README.md)
Module for feature scaling and standardization.

- Corresponding module name: FeatureScale

- Data Input: DTable, whose values are instances.
- Data Output: Transformed DTable.
- Model Output: Transform factors like min/max, mean/std.


#### 5. [Hetero Feature Binning](./feature/README.md)
With binning input data, calculates each column's iv and woe and transform data according to the binned information.

- Corresponding module name: HeteroFeatureBinning

- Data Input: DTable with y in guest and without y in host.
- Data Output: Transformed DTable.
- Model Output: iv/woe, split points, event counts, non-event counts etc. of each column.


#### 6. [OneHot Encoder](./feature/README.md)
Transfer a column into one-hot format.

- Corresponding module name: OneHotEncoder
- Data Input: Input DTable.
- Data Output: Transformed DTable with new headers.
- Model Output: Original header and feature values to new header map.


#### 7. [Hetero Feature Selection](./feature/README.md)
Provide 5 types of filters. Each filters can select columns according to user config.

- Corresponding module name: HeteroFeatureSelection
- Data Input: Input DTable.
- Model Input: If iv filters used, hetero_binning model is needed.
- Data Output: Transformed DTable with new headers and filtered data instance.
- Model Output: Whether left or not for each column.


#### 8. [Union](./statistic/union/README.md)
Combine multiple data tables into one. 

- Corresponding module name: Union
- Data Input: Input DTable(s).
- Data Output: one DTable with combined values from input DTables.


#### 9. [Hetero-LR](./linear_model/logistic_regression/README.md)
Build hetero logistic regression module through multiple parties.

- Corresponding module name: HeteroLR
- Data Input: Input DTable.
- Model Output: Logistic Regression model.


#### 10. [Local Baseline](./local_baseline/README.md)
Wrapper that runs sklearn Logistic Regression model with local data.

- Corresponding module name: LocalBaseline
- Data Input: Input DTable.
- Model Output: Logistic Regression.


#### 11. [Hetero-LinR](./linear_model/linear_regression/README.md)
Build hetero linear regression module through multiple parties.

- Corresponding module name: HeteroLinR
- Data Input: Input DTable.
- Model Output: Linear Regression model.


#### 12. [Hetero-Poisson](./linear_model/poisson_regression/README.md)
Build hetero poisson regression module through multiple parties.

- Corresponding module name: HeteroPoisson
- Data Input: Input DTable.
- Model Output: Poisson Regression model.


#### 13. [Homo-LR](./linear_model/logistic_regression/README.md)
Build homo logistic regression module through multiple parties.

- Corresponding module name: HomoLR
- Data Input: Input DTable.
- Model Output: Logistic Regression model.


#### 14. [Homo-NN](./nn/homo_nn/README.md)
Build homo neural network module through multiple parties.

- Corresponding module name: HomoNN
- Data Input: Input DTable.
- Model Output: Neural Network model.


#### 15. [Hetero Secure Boosting](./tree/README.md)
Build hetero secure boosting module through multiple parties.

Corresponding module name: HeteroSecureBoost

- Data Input: DTable, values are instances.
- Model Output: SecureBoost Model, consists of model-meta and model-param


#### 16. [Evaluation](./evaluation/README.md)
Output the model evaluation metrics for user.

- Corresponding module name: Evaluation


#### 17. [Hetero Pearson](./statistic/correlation/README.md)
Calculate hetero correlation of features from different parties.

- Corresponding module name: HeteroPearson


#### 18. [Hetero-NN](./nn/hetero_nn/README.md)
Build hetero neural network module.

- Corresponding module name: HeteroNN
- Data Input: Input DTable.
- Model Output: hetero neural network model.

### Secure Protocol
#### 1. [Homomorphic Encryption](./secureprotol/README.md)

- Paillier
- Affine Homomorphic Encryption
- IterativeAffine Homomorphic Encryption

#### 2. [SecretShare](./secureprotol/README.md)

- SPDZ

#### 3. [Diffne Hellman Key Exchange](./secureprotol/README.md)


#### 4. [RSA](./secureprotol/README.md)
