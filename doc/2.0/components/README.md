# Federated Machine Learning

[[中文](README.zh.md)]

FATE-ML includes implementation of many common machine learning
algorithms on federated learning. All modules are developed in a
decoupling modular approach to enhance scalability. Specifically, we
provide:

1. Federated Statistic: PSI, Union, Pearson Correlation, etc.
2. Federated Feature Engineering: Feature Sampling, Feature Binning,
   Feature Selection, etc.
3. Federated Machine Learning Algorithms: LR, GBDT, DNN
4. Model Evaluation: Binary | Multiclass | Regression | Clustering
   Evaluation
5. Secure Protocol: Provides multiple security protocols for secure
   multi-party computing and interaction between participants.

## Algorithm List

| Algorithm                                        | Module Name            | Description                                                                                                                        | Data Input                                    | Data Output                                                                | Model Input                   | Model Output |
|--------------------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------|-------------------------------|--------------|
| [PSI](psi.md)                                    | PSI                    | Compute intersect data set of multiple parties without leakage of difference set information. Mainly used in hetero scenario task. | input_data                                    | output_data                                                                |                               |              |
| [Sampling](sample.md)                            | Sample                 | Federated Sampling data so that its distribution become balance in each party.This module supports local and federation scenario.  | input_data                                    | output_data                                                                |                               |              |
| [Data Split](data_split.md)                      | DataSplit              | Split one data table into 3 tables by given ratio or count, this module supports local and federation scenario                     | input_data                                    | train_output_data, validate_output_data, test_output_data                  |                               |              |
| [Feature Scale](feature_scale.md)                | FeatureScale           | module for feature scaling and standardization.                                                                                    | train_data, test_data                         | train_output_data, test_output_data                                        | input_model                   | output_model |
| [Data Statistics](statistics.md)                 | Statistics             | This component will do some statistical work on the data, including statistical mean, maximum and minimum, median, etc.            | input_data                                    | output_data                                                                |                               | output_model |
| [Hetero Feature Binning](feature_binning.md)     | HeteroFeatureBinning   | With binning input data, calculates each column's iv and woe and transform data according to the binned information.               | train_data, test_data                         | train_output_data, test_output_data                                        | input_model                   | output_model |
| [Hetero Feature Selection](feature_selection.md) | HeteroFeatureSelection | Provide 3 types of filters. Each filters can select columns according to user config                                               | train_data, test_data                         | train_output_data, test_output_data                                        | input_models, input_model     | output_model |
| [Coordinated-LR](logistic_regression.md)         | CoordinatedLR          | Build hetero logistic regression model through multiple parties.                                                                   | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas | input_model, warm_start_model | output_model |
| [Coordinated-LinR](linear_regression.md)         | CoordinatedLinR        | Build hetero linear regression model through multiple parties.                                                                     | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas | input_model, warm_start_model | output_model |
| [Homo-LR](logistic_regression.md)                | HomoLR                 | Build homo logistic regression model through multiple parties.                                                                     | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas | input_model, warm_start_model | output_model |
| [Homo-NN](homo_nn.md)                            | HomoNN                 | Build homo neural network model through multiple parties.                                                                          | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas | input_model, warm_start_model | output_model |
| [Hetero Secure Boosting](ensemble.md)            | HeteroSecureBoost      | Build hetero secure boosting model through multiple parties                                                                        | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas | input_model, warm_start_model | output_model |
| [Evaluation](evaluation.md)                      | Evaluation             | Output the model evaluation metrics for user.                                                                                      | input_data                                    |                                                                            |                               |              |
| [Union](union.md)                                | Union                  | Combine multiple data tables into one.                                                                                             | input_data_list                               | output_data                                                                |                               |              |

## Secure Protocol

- [Encrypt](secureprotol.md#encrypt)
    - [Paillier encryption](secureprotol.md#paillier-encryption)
    - [RSA encryption](secureprotol.md#rsa-encryption)
- [Hash](secureprotol.md#hash-factory)
- [Diffie Hellman Key Exchange](secureprotol.md#diffie-hellman-key-exchange)
