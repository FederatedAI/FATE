Federated Machine Learning
==========================
[`中文`_]

.. _中文: README_zh.rst

FederatedML includes implementation of many common machine learning algorithms on federated learning. All modules are developed in a decoupling modular approach to enhance scalability. Specifically, we provide:

1. Federated Statistic: PSI, Union, Pearson Correlation, etc.

2. Federated Feature Engineering: Feature Sampling, Feature Binning, Feature Selection, etc.

3. Federated Machine Learning Algorithms: LR, GBDT, DNN, TransferLearning, which support Heterogeneous and Homogeneous styles.

4. Model Evaluation: Binary | Multiclass | Regression Evaluation, Local vs Federated Comparison.

5. Secure Protocol: Provides multiple security protocols for secure multi-party computing and interaction between participants.

.. image:: ../doc/images/federatedml_structure.png
   :width: 800
   :align: center
   :alt: federatedml structure


Algorithm List
---------------

.. list-table:: Algorithm
   :widths: 10 10 40 10 10 10 10
   :header-rows: 1

   * - Algorithm
     - Module Name
     - Description
     - Data Input
     - Data Output
     - Model Input
     - Model Output

   * - `DataIO`_
     - DataIO
     - This component transforms user-uploaded date into Instance object.
     - Table, values are raw data.
     - Transformed Table, values are data instance define in `federatedml/feature/instance.py`
     -
     - DataIO Model

   * - `Intersect`_
     - Intersection
     - Compute intersect data set of multiple parties without leakage of difference set information. Mainly used in hetero scenario task.
     - Table.
     - Table with only common instance keys.
     -
     - Intersect Model

   * - `Federated Sampling`_
     - FederatedSample
     - Federated Sampling data so that its distribution become balance in each party.This module supports standalone and federated versions.
     - Table
     - Table of sampled data; both random and stratified sampling methods are supported.
     -
     -

   * - `Feature Scale`_
     - FeatureScale
     - Module for feature scaling and standardization.
     - Table，values are instances.
     - Transformed Table.
     - Transform factors like min/max, mean/std.
     -

   * - `Hetero Feature Binning`_
     - Hetero Feature Binning
     - With binning input data, calculates each column's iv and woe and transform data according to the binned information.
     - Table, values are instances.
     - Transformed Table.
     -
     - iv/woe, split points, event counts, non-event counts etc. of each column.

   * - `OneHot Encoder`_
     - OneHotEncoder
     - Transfer a column into one-hot format.
     - Table, values are instances.
     - Transformed Table with new header.
     -
     - Feature-name mapping between original header and new header.

   * - `Hetero Feature Selection`_
     - HeteroFeatureSelection
     - Provide 5 types of filters. Each filters can select columns according to user config
     - Table
     - Transformed Table with new header and filtered data instance.
     - If iv filters used, hetero_binning model is needed.
     - Whether each column is filtered.

   * - `Union`_
     - Union
     - Combine multiple data tables into one.
     - Tables.
     - Table with combined values from input Tables.
     -
     -

   * - `Hetero-LR`_
     - HeteroLR
     - Build hetero logistic regression module through multiple parties.
     - Table, values are instances
     - Table, values are instances.
     -
     - Logistic Regression Model, consists of model-meta and model-param.

   * - `Local Baseline`_
     - LocalBaseline
     - Wrapper that runs sklearn Logistic Regression model with local data.
     - Table, values are instances.
     - Table, values are instances.
     -
     -

   * - `Hetero-LinR`_
     - HeteroLinR
     - Build hetero linear regression module through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Linear Regression Model, consists of model-meta and model-param.

   * - `Hetero-Poisson`_
     - HeteroPoisson
     - Build hetero poisson regression module through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Poisson Regression Model, consists of model-meta and model-param.

   * - `Homo-LR`_
     - HomoLR
     - Build homo logistic regression module through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Logistic Regression Model, consists of model-meta and model-param.

   * - `Homo-NN`_
     - HomoNN
     - Build homo neural network module through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Neural Network Model, consists of model-meta and model-param.

   * - `Hetero Secure Boosting`_
     - HeteroSecureBoost
     - Build hetero secure boosting module through multiple parties
     - Table, values are instances.
     - Table, values are instances.
     -
     - SecureBoost Model, consists of model-meta and model-param.

   * - `Evaluation`_
     - Evaluation
     - Output the model evaluation metrics for user.
     - Table(s), values are instances.
     -
     -
     -

   * - `Hetero Pearson`_
     - HeteroPearson
     - Calculate hetero correlation of features from different parties.
     - Table, values are instances.
     -
     -
     -

   * - `Hetero-NN`_
     - HeteroNN
     - Build hetero neural network module.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Hetero Neural Network Model, consists of model-meta and model-param.

   * - `Homo Secure Boosting`_
     - HomoSecureBoost
     - Build homo secure boosting module through multiple parties
     - Table, values are instances.
     - Table, values are instances.
     -
     - SecureBoost Model, consists of model-meta and model-param.

   * - `Homo OneHot Encoder`_
     - HomoOneHotEncoder
     - Build homo onehot encoder module through multiple parties.
     - Table, values are instances.
     - Transformed Table with new header.
     -
     - Feature-name mapping between original header and new header.

   * - `Data Split`_
     - Data Split
     - Split one data table into 3 tables by given ratio or count
     - Table, values are instances.
     - 3 Tables, values are instance.
     -
     -

   * - `Column Expand`_
     - Column Expand
     - Add arbitrary number of columns with user-provided values.
     - Table, values are raw data.
     - Transformed Table with added column(s) and new header.
     -
     - Column Expand Model

.. _DataIO: util/README.rst
.. _Intersect: statistic/intersect/README.rst
.. _Federated Sampling: feature/README.rst
.. _Feature Scale: feature/README.rst
.. _Hetero Feature Binning: feature/README.rst
.. _OneHot Encoder: feature/README.rst
.. _Hetero Feature Selection: feature/README.rst
.. _Union: statistic/union/README.rst
.. _Hetero-LR: linear_model/logistic_regression/README.rst
.. _Local Baseline: local_baseline/README.rst
.. _Hetero-LinR: linear_model/linear_regression/README.rst
.. _Hetero-Poisson: linear_model/poisson_regression/README.rst
.. _Homo-LR: linear_model/logistic_regression/README.rst
.. _Homo-NN: nn/homo_nn/README.rst
.. _Hetero Secure Boosting: tree/README.rst
.. _Evaluation: evaluation/README.rst
.. _Hetero Pearson: statistic/correlation/README.rst
.. _Hetero-NN: nn/hetero_nn/README.rst
.. _Homo Secure Boosting: tree/README.rst
.. _Data Split: model_selection/data_split/README.rst
.. _Homo OneHot Encoder: feature/README.rst
.. _Column Expand: feature/README.rst


Secure Protocol
---------------

* `Encrypt`_

   - `Paillier encryption`_
   - `Affine Homomorphic Encryption`_
   - `IterativeAffine Homomorphic Encryption`_
   - `RSA encryption`_
   - `Fake encryption`_

* `Encode`_

* `Diffne Hellman Key Exchange`_

* `SecretShare MPC Protocol(SPDZ)`_


.. _Encrypt: secureprotol/README.rst#encrypt
.. _Paillier encryption: secureprotol/README.rst#paillier-encryption
.. _Affine Homomorphic Encryption: secureprotol/README.rst#affine-homomorphic-encryption
.. _IterativeAffine Homomorphic Encryption: secureprotol/README.rst#iterativeaffine-homomorphic-encryption
.. _RSA encryption: secureprotol/README.rst#rst-encryption
.. _Fake encryption: secureprotol/README.rst#fake-encryption
.. _Encode: secureprotol/README.rst#encode
.. _Diffne Hellman Key Exchange: secureprotol/README.rst#diffne-hellman-key-exchange
.. _SecretShare MPC Protocol(SPDZ): secureprotol/README.rst#secretshare-mpc-protocol-spdz



Params
-------

.. automodule:: federatedml.param
   :autosummary:
   :members:
