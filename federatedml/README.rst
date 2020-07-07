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
   :align: center
   :alt: federatedml structure


Alogorithm List
---------------


+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| Algorithm                    | Module Name               | Description           | Data Input | Data Output    | Model Input | Model Output |
+==============================+===========================+=======================+============+================+=============+==============+
| `DataIO`_                    | DataIO                    | This component is     | DTable,    | Transformed    |             |              |
|                              |                           | typically the first   | values are | DTable, values |             |              |
|                              |                           | component of a        | raw data.  | are data       |             |              |
|                              |                           | modeling task. It     |            | instance       |             |              |
|                              |                           | will transform user-  |            | define in f    |             |              |
|                              |                           | uploaded date into    |            | ederatedml/    |             |              |
|                              |                           | Instance object which |            | feature/ins    |             |              |
|                              |                           | can be used for the   |            | tance.py       |             |              |
|                              |                           | following components. |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Intersect`_                 | Intersection              | Compute intersect     | DTable     | DTable which   |             |              |
|                              |                           | data set of two       |            | keys are       |             |              |
|                              |                           | parties without       |            | occurred in    |             |              |
|                              |                           | leakage of difference |            | both parties.  |             |              |
|                              |                           | set information.      |            |                |             |              |
|                              |                           | Mainly used in hetero |            |                |             |              |
|                              |                           | scenario task.        |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Federated Sampling`_        | FederatedSample           | Federated Sampling    | DTable     | the sampled    |             |              |
|                              |                           | data so that its      |            | data, supports |             |              |
|                              |                           | distribution become   |            | both random    |             |              |
|                              |                           | balance in each       |            | and stratified |             |              |
|                              |                           | party.This module     |            | sampling.      |             |              |
|                              |                           | support both          |            |                |             |              |
|                              |                           | federated and         |            |                |             |              |
|                              |                           | standalone version.   |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Feature Scale`_             | FeatureScale              | Module for feature    | DTable,    | Transformed    |             | Transform    |
|                              |                           | scaling and           | whose      | DTable.        |             | factors like |
|                              |                           | standardization.      | values are |                |             | min/max,     |
|                              |                           |                       | instances. |                |             | mean/std.    |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero Feature Binning`_    | HeteroFeatureBinning      | With binning input    | DTable     | Transformed    |             | iv/woe,      |
|                              |                           | data, calculates each | with y in  | DTable.        |             | split        |
|                              |                           | column's iv and woe   | guest and  |                |             | points,      |
|                              |                           | and transform data    | without y  |                |             | event        |
|                              |                           | according to the      | in host.   |                |             | counts, non- |
|                              |                           | binned information.   |            |                |             | event counts |
|                              |                           |                       |            |                |             | etc. of each |
|                              |                           |                       |            |                |             | column.      |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `OneHot Encoder`_            | OneHotEncoder             | Transfer a column     | Input      | Transformed    |             | Original     |
|                              |                           | into one-hot format.  | DTable.    | DTable with    |             | header and   |
|                              |                           |                       |            | new headers.   |             | feature      |
|                              |                           |                       |            |                |             | values to    |
|                              |                           |                       |            |                |             | new header   |
|                              |                           |                       |            |                |             | map.         |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero Feature Selection`_  | HeteroFeatureSelection    | Provide 5 types of    | Input      | Transformed    | If iv       | Whether left |
|                              |                           | filters. Each filters | DTable.    | DTable with    | filters     | or not for   |
|                              |                           | can select columns    |            | new headers    | used, heter | each column. |
|                              |                           | according to user     |            | and filtered   | o_binning   |              |
|                              |                           | config.               |            | data instance. | model is    |              |
|                              |                           |                       |            |                | needed.     |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Union`_                     | Union                     | Combine multiple data | Input      | one DTable     |             |              |
|                              |                           | tables into one.      | DTable(s). | with combined  |             |              |
|                              |                           |                       |            | values from    |             |              |
|                              |                           |                       |            | input DTables. |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero-LR`_                 | HeteroLR                  | Build hetero logistic | Input      |                |             | Logistic     |
|                              |                           | regression module     | DTable.    |                |             | Regression   |
|                              |                           | through multiple      |            |                |             | model.       |
|                              |                           | parties.              |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Local Baseline`_            | LocalBaseline             | Wrapper that runs     | Input      |                |             | Logistic     |
|                              |                           | sklearn Logistic      | DTable.    |                |             | Regression.  |
|                              |                           | Regression model with |            |                |             | model.       |
|                              |                           | local data.           |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero-LinR`_               | HeteroLinR                | Build hetero linear   | Input      | \              | \           | Linear       |
|                              |                           | regression module     | DTable.    |                |             | Regression   |
|                              |                           | through multiple      |            |                |             | model.       |
|                              |                           | parties.              |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero-Poisson`_            | HeteroPoisson             | Build hetero poisson  | Input      | \              | \           | Poisson      |
|                              |                           | regression module     | DTable.    |                |             | Regression   |
|                              |                           | through multiple      |            |                |             | model.       |
|                              |                           | parties.              |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Homo-LR`_                   | HomoLR                    | Build homo logistic   | Input      | \              | \           | Logistic     |
|                              |                           | regression module     | DTable.    |                |             | Regression   |
|                              |                           | through multiple      |            |                |             | model.       |
|                              |                           | parties.              |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Homo-NN`_                   | HomoNN                    | Build homo neural     | Input      | \              | \           | Neural       |
|                              |                           | network module        | Dtable.    |                |             | Network      |
|                              |                           | through multiple      |            |                |             | model.       |
|                              |                           | parties.              |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero Secure Boosting`_    | HeteroSecureBoost         | Build hetero secure   | DTable,    | \              | \           | SecureBoost  |
|                              |                           | boosting module       | values are |                |             | Model,       |
|                              |                           | through multiple      | instances. |                |             | consists of  |
|                              |                           | parties.              |            |                |             | model-meta   |
|                              |                           |                       |            |                |             | and model-   |
|                              |                           |                       |            |                |             | param        |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Evaluation`_                | Evaluation                |                       |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero Pearson`_            | HeteroPearson             |                       |            |                |             |              |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Hetero-NN`_                 | HeteroNN                  | Build hetero neural   | Input      | \              | \           | Model        |
|                              |                           | network module.       | Dtable.    |                |             | Output:      |
|                              |                           |                       |            |                |             | heero neural |
|                              |                           |                       |            |                |             | network      |
|                              |                           |                       |            |                |             | model.       |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+
| `Homo Secure Boosting`_      | HomoSecureBoost           | Build homo secure     | DTable,    | \              | \           | SecureBoost  |
|                              |                           | boosting module       | values are |                |             | Model,       |
|                              |                           | through multiple      | instances. |                |             | consists of  |
|                              |                           | parties.              |            |                |             | model-meta   |
|                              |                           |                       |            |                |             | and model-   |
|                              |                           |                       |            |                |             | param        |
+------------------------------+---------------------------+-----------------------+------------+----------------+-------------+--------------+




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




.. toctree::
   :maxdepth: 2

   util/README
   statistic/intersect/README
   feature/README
   statistic/union/README
   linear_model/logistic_regression/README
   local_baseline/README
   linear_model/linear_regression/README
   linear_model/poisson_regression/README
   nn/homo_nn/README
   tree/README
   evaluation/README
   statistic/correlation/README
   nn/hetero_nn/README
   model_selection/stepwise/README




Secure Protocol
---------------


.. toctree::
   :maxdepth: 2

   secureprotol/README
   

