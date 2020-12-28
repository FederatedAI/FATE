Federated Machine Learning
==========================
[`中文`_]

.. _中文: README_zh.rst

FederatedML includes implementation of many common machine learning algorithms on federated learning. All modules are developed in a decoupling modular approach to enhance scalability. Specifically, we provide:

1. Federated Statistic: PSI, Union, Pearson Correlation, etc.

2. Federated Feature Engineering: Feature Sampling, Feature Binning, Feature Selection, etc.

3. Federated Machine Learning Algorithms: LR, GBDT, DNN, TransferLearning, which support Heterogeneous and Homogeneous styles.

4. Model Evaluation: Binary | Multiclass | Regression | Clustering Evaluation, Local vs Federated Comparison.

5. Secure Protocol: Provides multiple security protocols for secure multi-party computing and interaction between participants.

.. image:: ../../doc/images/federatedml_structure.png
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

   * - Reader
     - Reader
     - This component loads and transforms data from storage engine so that data is compatible with FATE computing engine
     - Original Data
     - Transformed Data
     -
     -

   * - `DataIO`_
     - DataIO
     - This component transforms user-uploaded date into Instance object.
     - Table, values are raw data.
     - Transformed Table, values are data instance defined `here <./feature/instance.py>`_
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
     - module for feature scaling and standardization.
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
     - iv/woe, split points, event count, non-event count etc. of each column.

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
     - Build hetero logistic regression model through multiple parties.
     - Table, values are instances
     - Table, values are instances.
     -
     - Logistic Regression Model, consists of model-meta and model-param.

   * - `Local Baseline`_
     - LocalBaseline
     - Wrapper that runs sklearn(scikit-learn) Logistic Regression model with local data.
     - Table, values are instances.
     - Table, values are instances.
     -
     -

   * - `Hetero-LinR`_
     - HeteroLinR
     - Build hetero linear regression model through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Linear Regression Model, consists of model-meta and model-param.

   * - `Hetero-Poisson`_
     - HeteroPoisson
     - Build hetero poisson regression model through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Poisson Regression Model, consists of model-meta and model-param.

   * - `Homo-LR`_
     - HomoLR
     - Build homo logistic regression model through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Logistic Regression Model, consists of model-meta and model-param.

   * - `Homo-NN`_
     - HomoNN
     - Build homo neural network model through multiple parties.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Neural Network Model, consists of model-meta and model-param.

   * - `Hetero Secure Boosting`_
     - HeteroSecureBoost
     - Build hetero secure boosting model through multiple parties
     - Table, values are instances.
     - Table, values are instances.
     -
     - SecureBoost Model, consists of model-meta and model-param.

   * - `Hetero Fast Secure Boosting`_
     - HeteroFastSecureBoost
     - Build hetero secure boosting model through multiple parties in layered/mix manners.
     - Table, values are instances.
     - Table, values are instances.
     -
     - FastSecureBoost Model, consists of model-meta and model-param.

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
     - Build hetero neural network model.
     - Table, values are instances.
     - Table, values are instances.
     -
     - Hetero Neural Network Model, consists of model-meta and model-param.

   * - `Homo Secure Boosting`_
     - HomoSecureBoost
     - Build homo secure boosting model through multiple parties
     - Table, values are instances.
     - Table, values are instances.
     -
     - SecureBoost Model, consists of model-meta and model-param.

   * - `Homo OneHot Encoder`_
     - HomoOneHotEncoder
     - Build homo onehot encoder model through multiple parties.
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

   * - `Secure Information Retrieval`_
     - Secure Information Retrieval
     - Securely retrieves information from host through oblivious transfer
     - Table, values are instance
     - Table, values are instance
     -
     -

   * - `Hetero Federated Transfer Learning`_
     - Hetero FTL
     - Build Hetero FTL Model Between 2 party
     - Table, values are instance
     -
     -
     - Hetero FTL Model

   * - `Hetero KMeans`_
     - Hetero KMeans
     - Build Hetero KMeans model through multiple parties
     - Table, values are instance
     - Table, values are instance; Arbier outputs 2 Tables
     -
     - Hetero KMeans Model

   * - `PSI`_
     - PSI module
     - Compute PSI value of features between two table
     - Table, values are instance
     -
     -
     - PSI Results

   * - `Data Statistics`_
     - Data Statistics
     - This component will do some statistical work on the data, including statistical mean, maximum and minimum, median, etc.
     - Table, values are instance
     - Table
     -
     - Statistic Result

   * - `Scorecard`_
     - Scorecard
     - Scale predict score to credit score by given scaling parameters
     - Table, values are predict score
     - Table, values are score results
     -
     -

   * - `Secret Sharing Sum`_
     - Secret Sharing Sum
     - This component will sum multiple privacy values without exposing data
     - Table, values are addend
     - Table, values are sum results
     -
     -



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
.. _Hetero Secure Boosting: ensemble/README.rst
.. _Evaluation: evaluation/README.rst
.. _Hetero Pearson: statistic/correlation/README.rst
.. _Hetero-NN: nn/hetero_nn/README.rst
.. _Homo Secure Boosting: ensemble/README.rst
.. _Data Split: model_selection/data_split/README.rst
.. _Homo OneHot Encoder: feature/README.rst
.. _Column Expand: feature/README.rst
.. _Secure Information Retrieval: secure_information_retrieval
.. _Hetero KMeans: unsupervised_learning/kmeans/README.rst
.. _Data Statistics: statistic/README.rst
.. _PSI: statistic/psi/README.rst
.. _Hetero Federated Transfer Learning: transfer_learning/hetero_ftl/README.rst
.. _Hetero Fast Secure Boosting: ensemble/README.rst
.. _Scorecard: statistic/scorecard/README.rst
.. _Secret Sharing Sum: secret_sharing_sum/README.rst


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

* `Oblivious Transfer`_

* `Verifiable Secret Sharing`_


.. _Encrypt: secureprotol/README.rst#encrypt
.. _Paillier encryption: secureprotol/README.rst#paillier-encryption
.. _Affine Homomorphic Encryption: secureprotol/README.rst#affine-homomorphic-encryption
.. _IterativeAffine Homomorphic Encryption: secureprotol/README.rst#iterativeaffine-homomorphic-encryption
.. _RSA encryption: secureprotol/README.rst#rst-encryption
.. _Fake encryption: secureprotol/README.rst#fake-encryption
.. _Encode: secureprotol/README.rst#encode
.. _Diffne Hellman Key Exchange: secureprotol/README.rst#diffne-hellman-key-exchange
.. _SecretShare MPC Protocol(SPDZ): secureprotol/README.rst#secretshare-mpc-protocol-spdz
.. _Oblivious Transfer: secureprotol/README.rst#oblivious-transfer
.. _Verifiable Secret Sharing: secureprotol/README.rst#verifiable-secret-sharing


Params
-------

.. automodule:: federatedml.param
   :autosummary:
   :members:
