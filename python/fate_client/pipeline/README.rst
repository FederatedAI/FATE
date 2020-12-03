FATE Pipeline
=============

Pipeline is a high-level python API that allows user to design, start,
and query FATE jobs in a sequential manner. FATE Pipeline is designed to
be user-friendly and consistent in behavior with FATE command line
tools. User can customize job workflow by adding components to pipeline
and then initiate a job with one call. In addition, Pipeline provides
functionality to run prediction and query information after fitting a
pipeline. Run the `mini demo <./demo/pipeline-mini-demo.py>`__ to have a
taste of how FATE Pipeline works. Default values of party ids and work
mode may need to be modified depending on the deployment setting.

.. code:: bash

   python pipeline-mini-demo.py config.yaml

For more pipeline demo, please refer to
`examples <../../../examples/pipeline>`__.

A FATE Job is A Directed Acyclic Graph
--------------------------------------

A FATE job is a dag that consists of algorithm component nodes. FATE pipeline provides
easy-to-use tools to configure order and setting of the tasks.

FATE is written in a modular style. Modules are designed to have input
and output data and model. Therefore two modules are connected when
output of one module is set to be the input of another module. By tracing
how one data set is processed through FATE modules, we can see that a
FATE job is in fact formed by a sequence of sub-tasks. For example, in
the `mini demo <./demo/pipeline-mini-demo.py>`__ above, guest’s data is
first read in by ``Reader``, then loaded into ``DataIO``. Overlapping
ids between guest and host are then found by running data through
``Intersection``. Finally, ``HeteroLR`` model is fit on the data. Each
listed modules run a small task with the data, and together they
constitute a model training job.

Beyond the given mini demo, a job may include multiple data sets and
models. For more pipeline examples, please refer to `examples <../../../examples/pipeline>`__.

Install Pipeline
----------------

Pipeline CLI
~~~~~~~~~~~~

After successfully installed FATE Client, user needs to configure server information and log directory for Pipeline.
Pipeline provides a command line tool for quick setup. Run the following
command for more information.

.. code:: bash

   pipeline init --help


Interface of Pipeline
---------------------

Component
~~~~~~~~~

FATE modules are wrapped into ``component`` in Pipeline API. Each
component can take in and output ``Data`` and ``Model``. Parameters of
components can be set conveniently at the time of initialization.
Unspecified parameters will take default values. All components have a
``name``, which can be arbitrarily set. A component’s name is its
identifier, and so it must be unique within a pipeline. We suggest that
each component name includes a numbering as suffix for easy tracking.

Components each may have input and/or output ``Data`` and/or ``Model``.
For details on how to use component, please refer to this
`guide <./component/README.rst>`__.

An example of initializing a component with specified parameter values:

.. code:: python

   hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10,
                          early_stopping_rounds=2, validation_freqs=2)

Input
~~~~~~

`Input <./component/README.rst>`__ encapsulates all input of a component, including
``Data`` and ``Model`` input. To access ``input`` of a component,
reference its ``input`` attribute:

.. code:: python

   input_all = dataio_0.input

Output
~~~~~~

`Output <./component/README.rst>`__ encapsulates all output result of a component, including
``Data`` and ``Model`` output. To access ``Output`` from a component,
reference its ``output`` attribute:

.. code:: python

   output_all = dataio_0.output

Data
~~~~

``Data`` wraps all data-type input and output of components.
FATE Pipeline includes five types of ``data``, each is used for different scenario.
For more information, please refer `here <./component/README.rst>`__.

Model
~~~~~

``Model`` defines model input and output of components. Similar to ``Data``, the two
types of ``models`` are used for different purposes.
For more information, please refer `here <./component/README.rst>`__.

Build A Pipeline
----------------

Below is a general guide to building a pipeline. Please refer to `mini
demo <./demo/pipeline-mini-demo.py>`__ for an explained demo.

Once initialized a pipeline, job participants and initiator should be
specified. Below is an example of initial setup of a pipeline:

.. code:: python

   pipeline = PipeLine()
   pipeline.set_initiator(role='guest', party_id=9999)
   pipeline.set_roles(guest=9999, host=10000, arbiter=10000)

``Reader`` is required to read in data source so that other component(s)
can process data. Define a ``Reader`` component:

.. code:: python

   reader_0 = Reader(name="reader_0")

In most cases, ``DataIO`` follows ``Reader`` to transform data into
DataInstance format, which can then be used for data engineering and
model training. Some components (such as ``Union`` and ``Intersection``)
can run directly on non-DataInstance tables.

All pipeline components can be configured individually for different
roles by setting ``get_party_instance``. For instance, ``DataIO``
component can be configured specifically for guest like this:

.. code:: python

   dataio_0 = DataIO(name="dataio_0")
   guest_component_instance = dataio_0.get_party_instance(role='guest', party_id=9999)
   guest_component_instance.component_param(with_label=True, output_format="dense")

To include a component in a pipeline, use ``add_component``. To add the
``DataIO`` component to the previously created pipeline, try this:

.. code:: python

   pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))


Build Fate NN Model In Keras Style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In pipeline, you can build NN structures in a Keras style. Take Homo-NN as an example:

First, import Keras and define your nn structures:

.. code:: python

    from tensorflow.keras import optimizers
    from tensorflow.keras.layers import Dense

    layer_0 = Dense(units=6, input_shape=(10,), activation="relu")
    layer_1 = Dense(units=1, activation="sigmoid")

Then, add nn layers into Homo-NN model like using Sequential class in Keras:

.. code:: python

    from pipeline.component.homo_nn import HomoNN

    # set parameter
    homo_nn_0 = HomoNN(name="homo_nn_0", max_iter=10, batch_size=-1, early_stop={"early_stop": "diff", "eps": 0.0001})
    homo_nn_0.add(layer_0)
    homo_nn_0.add(layer_1)

Set optimizer and compile Homo-NN model:

.. code:: python

    homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=0.05), metrics=["Hinge", "accuracy", "AUC"],
                      loss="binary_crossentropy")

Add it to pipeline:

.. code:: python

    pipeline.add_component(homo_nn, data=Data(train_data=dataio_0.output.data))

Init Runtime JobParameters
--------------------------

To fit or predict, user needs to initialize the runtime environment, like 'backend' and 'work_mode',

.. code:: python

    from pipeline.runtime.entity import JobParameters
    job_parameters = JobParameters(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

Run A Pipeline
--------------

Having added all components, user needs to first compile pipeline before
running the designed job. After compilation, the pipeline can then be fit(run
train job) with appropriate ``Backend`` and ``WorkMode``.

.. code:: python

   pipeline.compile()
   pipeline.fit(job_parameters)

Query on Tasks
--------------

FATE Pipeline provides API to query component information,
including data, model, and summary. All query API have matching name to
`FlowPy <../flow_sdk>`__, while Pipeline retrieves and returns
query result directly to user.

.. code:: python

   summary = pipeline.get_component("hetero_lr_0").get_summary()

Deploy Components
-----------------

Once fitting pipeline completes, prediction can be run on new data set.
Before prediction, necessary components need to be first deployed. This
step marks selected components to be used by prediction pipeline.

.. code:: python

   # deploy select components
   pipeline.deploy_component([dataio_0, hetero_lr_0])
   # deploy all components
   # note that Reader component cannot be deployed. Always deploy pipeline with Reader by specified component list.
   pipeline.deploy_component()

Predict with Pipeline
---------------------

First, initiate a new pipeline, then specify data source used for
prediction.

.. code:: python

   predict_pipeline = PipeLine()
   predict_pipeline.add_component(reader_0)
   predict_pipeline.add_component(pipeline,
                                  data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}))

Prediction can then be initiated on the new pipeline.

.. code:: python

   predict_pipeline.predict(job_parameters)

In addition, since pipeline is modular, user may add new components to
the original pipeline before running prediction.

.. code:: python

   predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.hetero_lr_0.output.data))
   predict_pipeline.predict(job_parameters)


Save and Recovery of Pipeline
-----------------------------

To save a pipeline, just use **dump** interface.

.. code:: python

   pipeline.dump("pipeline_saved.pkl")

To restore a pipeline, use **load_model_from_file** interface.

.. code:: python

   from pipeline.backend.pipeline import PineLine
   PipeLine.load_model_from_file("pipeline_saved.pkl")

Summary Info of Pipeline
-------------------------

To get the details of a pipeline, use **describe** interface, which prints the "create time"
fit or predict state and the constructed dsl if exists.

.. code:: python

   pipeline.describe()

Upload Data
-----------

Pipeline provides functionality to upload local data table. Please refer
to `upload demo <./demo/pipeline-upload.py>`__ for a quick example. Note
that uploading data can be added all at once, and the pipeline used to
perform upload can be either training or prediction pipeline (or, a
separate pipeline as in the demo).

Pipeline vs. CLI
----------------

In the past versions, user interacts with FATE through command line
interface, often with manually configured conf and dsl json files. Manual
configuration can be tedious and error-prone. FATE Pipeline forms task
configure files automatically at compilation, allowing quick experiment
with task design.
