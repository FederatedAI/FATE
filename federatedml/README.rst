Federated Machine Learning
==========================
[`中文`_]

.. _中文: README_zh.rst

FederatedML includes implementation of many common machine learning algorithms on federated learning. All modules are developed in a decoupling modular approach to enhance scalability. Specifically, we provide:

1. Federated Statistic: PSI, Union, Pearson Correlation, etc.

2. Federated Feature Engineering: Feature Sampling, Feature Binning, Feature Selection, etc.

3. Federated Machine Learning Algorithms: LR, GBDT, DNN, TransferLearning, which support Heterogeneous and Homogeneous styles.

4. Model Evaluation: Binary|Multiclass|Regression Evaluation, Local vs Federated Comparison.

5. Secure Protocol: Provides multiple security protocols for secure multi-party computing and interaction between participants.

.. image:: ../doc/images/federatedml_structure.png
   :width: 800
   :alt: federatedml structure

Algorithm List
--------------

+-----------------+------------------+-----------------------+------------+----------------+-------------+--------------+
| Algorithm       | Module Name      | Description           | Data Input | Data Output    | Model Input | Model Output |
+=================+==================+=======================+============+================+=============+==============+
| `DataIO`_       | DataIO           | This component is     | DTable,    | Transformed    |             |              |
|                 |                  | typically the first   | values are | DTable, values |             |              |
|                 |                  | component of a        | raw data.  | are data       |             |              |
|                 |                  | modeling task. It     |            | instance       |             |              |
|                 |                  | will transform user-  |            | define in f    |             |              |
|                 |                  | uploaded date into    |            | ederatedml/    |             |              |
|                 |                  | Instance object which |            | feature/ins    |             |              |
|                 |                  | can be used for the   |            | tance.py       |             |              |
|                 |                  | following components. |            |                |             |              |
+-----------------+------------------+-----------------------+------------+----------------+-------------+--------------+
| `Intersect`_    | Intersection     | Compute intersect     | DTable     | DTable which   |             |              |
|                 |                  | data set of two       |            | keys are       |             |              |
|                 |                  | parties without       |            | occurred in    |             |              |
|                 |                  | leakage of difference |            | both parties.  |             |              |
|                 |                  | set information.      |            |                |             |              |
|                 |                  | Mainly used in hetero |            |                |             |              |
|                 |                  | scenario task.        |            |                |             |              |
+-----------------+------------------+-----------------------+------------+----------------+-------------+--------------+
| `Federated      | FederatedSample  | Federated Sampling    | DTable     | the sampled    |             |              |
| Sampling`_      |                  | data so that its      |            | data, supports |             |              |
|                 |                  | distribution become   |            | both random    |             |              |
|                 |                  | balance in each       |            | and stratified |             |              |
|                 |                  | party.This module     |            | sampling.      |             |              |
|                 |                  | support both          |            |                |             |              |
|                 |                  | federated and         |            |                |             |              |
|                 |                  | standalone version.   |            |                |             |              |
+-----------------+------------------+-----------------------+------------+----------------+-------------+--------------+



.. _DataIO: util/README.rst
.. _Intersect: statistic/intersect/README.md
.. _Federated Sampling: feature/README.md
