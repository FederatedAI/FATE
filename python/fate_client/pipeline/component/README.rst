Components
==========

Each ``Component`` wraps a `FederatedML <../../python/federatedml/README.rst>`__ ``Module``.
``Module`` implement machine learning algorithms on federated learning,
while ``Component`` provide convenient interface for easy model building.

Input & Output
--------------

Input
~~~~~

``Input`` encapsulates all upper stream input into a component in a FATE job.
There are two classes of ``input``: ``data`` and ``model``. Not all components have
both classes of input, and a component may accept only some types of a input class.

Here is an example to access a component's input:

.. code:: python

   from pipeline.component import DataIO
   dataio_0 = DataIO(name="dataio_0")
   input_all = dataio_0.input
   input_data = dataio_0.input.data
   input_model = dataio_0.input.model

Output
~~~~~~

Similar to ``Input``, ``Output`` encapsulates output ``data`` and ``model`` of component
in a FATE job. Not all components have both classes of outputs.

Here is an example to access a component's output:

.. code:: python

   from pipeline.component import DataIO
   dataio_0 = DataIO(name="dataio_0")
   output_all = dataio_0.output
   input_data = dataio_0.output.data
   input_model = dataio_0.output.model

Data
~~~~

In most cases, data sets are wrapped into ``data`` when being passed
between modules. For instance, in the `mini demo <../demo/pipeline-mini-demo.py>`_, data output of
``dataio_0`` is set as data input to ``intersection_0``.

.. code:: python

   pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))

For data sets used in different modeling stages (e.g., train & validate)
of the same component, additional keywords ``train_data``,
``validate_data`` and ``test_data`` are used to distinguish data sets.
Also from mini `mini demo <../demo/pipeline-mini-demo.py>`_, result from
``intersection_0`` and ``intersection_1`` are set as train and validate data
input to hetero logistic regression component, respectively.

.. code:: python

   pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data,
                                                 validate_data=intersection_1.output.data))

Another case of using keywords ``train_data``, ``validate_data``, and
``test_data`` is to use data output from ``DataSplit`` module, which always has three data outputs:
``train_data``, ``validate_data``, and ``test_data``.

.. code:: python

   pipeline.add_component(hetero_linr_1,
                          data=Data(test_data=hetero_data_split_0.output.data.test_data),
                          model=Model(model=hetero_linr_0))


Below lists all five types of ``data`` and whether ``Input`` and ``Output`` include them.

.. list-table:: Data
   :widths: 30 20 20 30
   :header-rows: 1

   * - Data Name
     - Input
     - Output
     - Use Case

   * - data
     - Yes
     - Yes
     - single data input or output

   * - train_data
     - Yes
     - Yes
     - model training; output of ``DataSplit`` component

   * - validate_data
     - Yes
     - Yes
     - model training with validate data; output of ``DataSplit`` component

   * - test_data
     - No
     - Yes
     - output of ``DataSplit`` component

   * - predict_data
     - Yes
     - No
     - model prediction

All input and output data of components need to be wrapepd into ``Data``
objects when being passed between components.
Here is a an example of chaining components with different types of data input and output:

.. code:: python

   from pipeline.backend.pipeline import Pipeline
   from pipeline.component import DataIO, Intersection, HeteroDataSplit, HeteroLR
   # initialize a pipeline
   pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest)
   # define all components
   dataio_0 = DataIO(name="dataio_0")
   data_split = HeteroDataSplit(name="data_split_0")
   hetero_lr_0 = HeteroLR(name="hetero_lr_0", max_iter=20)
   # chain together all components
   pipeline.add_component(reader_0)
   pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
   pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
   pipeline.add_component(hetero_data_split_0, data=Data(data=intersection_0.output.data))
   pipeline.add_component(hetero_lr_0, data=Data(train_data=hetero_data_split_0.output.data.train_data,
                                                 validate_data=hetero_data_split_0.output.data.test_data))

Model
~~~~~

There are two types of ``Model``: ``model`` and\ ``isometric_model``. When the current
component is of the same class as the previous component, if receiving
``model``, the current component will replicate all model parameters from
the previous component. When a model from previous component is used as
input but the current component is of different class from the previous component,
``isometric_model`` is used.

Check below for a case from mini demo, where ``model`` from ``dataio_0``
is passed to ``dataio_1``.

.. code:: python

   pipeline.add_component(dataio_1,
                          data=Data(data=reader_1.output.data),
                          model=Model(dataio_0.output.model))

Here is a case of using ``isometric model``. ``HeteroFeatureSelection`` uses
``isometric_model`` from ``HeteroFeatureBinning`` to select most
important features.

.. code:: python

   pipeline.add_component(hetero_feature_selection_0,
                          data=Data(data=intersection_0.output.data),
                          isometric_model=Model(hetero_feature_binning_0.output.model))

Please note that when using `stepwise` or `cross validation` method, components do
not have ``model`` output.

Parameter
~~~~~~~~~

Parameters of underlying module can be set for all job participants or per individual participant.

1. Parameters for all participants may be specified when defining a component:

.. code:: python

   from pipeline.component import DataIO
   dataio_0 = DataIO(name="dataio_0", input_format="dense", output_format="dense",
                     outlier_replace=False)

2. Parameters can be set for each party individually:

.. code:: python

   # set guest dataio_0 component parameters
   guest_dataio_0 = dataio_0.get_party_instance(role='guest', party_id=9999)
   guest_dataio_0.algorithm_param(with_label=True)
   # directly set host dataio_0 component parameters
   dataio_0.get_party_instance(role='host', party_id=10000).algorithm_param(with_label=False)

Component List
--------------

Below lists input and output elements of each component.

.. list-table:: Component
   :widths: 10 10 40 10 10 10 10
   :header-rows: 1

   * - Algorithm
     - Component Name
     - Description
     - Acceptable Input Data
     - Acceptable Output Data
     - Acceptable Input Model
     - Acceptable Output Model

   * - `Reader`_
     - Reader
     - This component is always the first component of a pipeline task(except for upload). It loads raw data from storage.
     - None
     - data
     - None
     - None

   * - `DataIO`_
     - DataIO
     - This component usually follows ``Reader``. It transforms user-uploaded date into Instance object.
     - data
     - data
     - model
     - model

   * - `Intersect`_
     - Intersection
     - Compute intersect data set of multiple parties without leakage of difference set information. Mainly used in hetero scenario task.
     - data
     - data
     - model
     - model

   * - `Federated Sampling`_
     - FederatedSample
     - Federated Sampling data so that its distribution become balance in each party.This module supports standalone and federated versions.
     - data
     - data
     - model
     - model

   * - `Feature Scale`_
     - FeatureScale
     - Feature scaling and standardization.
     - data
     - data
     - model
     - model

   * - `Hetero Feature Binning`_
     - Hetero Feature Binning
     - With binning input data, calculates each column's iv and woe and transform data according to the binned information.
     - data
     - data
     - model
     - model

   * - `OneHot Encoder`_
     - OneHotEncoder
     - Transfer a column into one-hot format.
     - data
     - data
     - model
     - model

   * - `Hetero Feature Selection`_
     - HeteroFeatureSelection
     - Provide 5 types of filters. Each filters can select columns according to user config
     - data
     - data
     - model; isometric model
     - model

   * - `Union`_
     - Union
     - Combine multiple data tables into one.
     - List[data]
     - data
     - model
     - model

   * - `Hetero-LR`_
     - HeteroLR
     - Build hetero logistic regression module through multiple parties.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Local Baseline`_
     - LocalBaseline
     - Wrapper that runs sklearn Logistic Regression model with local data.
     - train_data; validate_data; predict_data
     - data
     - None
     - None

   * - `Hetero-LinR`_
     - HeteroLinR
     - Build hetero linear regression module through multiple parties.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Hetero-Poisson`_
     - HeteroPoisson
     - Build hetero poisson regression module through multiple parties.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Homo-LR`_
     - HomoLR
     - Build homo logistic regression module through multiple parties.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Homo-NN`_
     - HomoNN
     - Build homo neural network module through multiple parties.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Hetero Secure Boosting`_
     - HeteroSecureBoost
     - Build hetero secure boosting module through multiple parties
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Evaluation`_
     - Evaluation
     - Output the model evaluation metrics for user.
     - data; List[data]
     - None
     - model
     - None

   * - `Hetero Pearson`_
     - HeteroPearson
     - Calculate hetero correlation of features from different parties.
     - data
     - None
     - model
     - model

   * - `Hetero-NN`_
     - HeteroNN
     - Build hetero neural network module.
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Homo Secure Boosting`_
     - HomoSecureBoost
     - Build homo secure boosting module through multiple parties
     - train_data; validate_data; predict_data
     - data
     - model
     - model

   * - `Homo OneHot Encoder`_
     - HomoOneHotEncoder
     - Build homo onehot encoder module through multiple parties.
     - data
     - data
     - model
     - model

   * - `Data Split`_
     - Data Split
     - Split one data table into 3 tables by given ratio or count
     - data
     - train_data & validate_data & test_data
     - None
     - None

   * - `Column Expand`_
     - Column Expand
     - Add arbitrary number of columns with user-provided values.
     - data
     - data
     - model
     - model

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


Params
-------

.. automodule:: pipeline/param
   :autosummary:
   :members:
