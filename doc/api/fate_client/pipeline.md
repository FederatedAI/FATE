# FATE Pipeline

Pipeline is a high-level python API that allows user to design, start,
and query FATE jobs in a sequential manner. FATE Pipeline is designed to
be user-friendly and consistent in behavior with FATE command line
tools. User can customize job workflow by adding components to pipeline
and then initiate a job with one call. In addition, Pipeline provides
functionality to run prediction and query information after fitting a
pipeline. Run the [mini
demo](../../../examples/pipeline/demo/pipeline-mini-demo.py) to have
a taste of how FATE Pipeline works. Default values of party ids and work
mode may need to be modified depending on the deployment setting.

```bash
python pipeline-mini-demo.py
```

For more pipeline demo, please refer to
[examples](../../../examples/pipeline).

## A FATE Job is A Directed Acyclic Graph

A FATE job is a dag that consists of algorithm component nodes. FATE
pipeline provides easy-to-use tools to configure order and setting of
the tasks.

FATE is written in a modular style. Modules are designed to have input
and output data and model. Therefore two modules are connected when
output of one module is set to be the input of another module. By
tracing how one data set is processed through FATE modules, we can see
that a FATE job is in fact formed by a sequence of sub-tasks. For
example, in the [mini
demo](../../../examples/pipeline/demo/pipeline-mini-demo.py) above,
guest’s data is first read in by `Reader`, then loaded into
`DataTransform`. Overlapping ids between guest and host are then found
by running data through `Intersection`. Finally, `HeteroLR` model is fit
on the data. Each listed modules run a small task with the data, and
together they constitute a model training job.

Beyond the given mini demo, a job may include multiple data sets and
models. For more pipeline examples, please refer to
[examples](../../../examples/pipeline).

## Install Pipeline

### Pipeline CLI

After successfully installed FATE Client, user needs to configure server
information and log directory for Pipeline. Pipeline provides a command
line tool for quick setup. Run the following command for more
information.

``` sourceCode bash
pipeline init --help
```

## Interface of Pipeline

### Component

FATE modules are wrapped into `component` in Pipeline API. Each
component can take in and output `Data` and `Model`. Parameters of
components can be set conveniently at the time of initialization.
Unspecified parameters will take default values. All components have a
`name`, which can be arbitrarily set. A component’s name is its
identifier, and so it must be unique within a pipeline. We suggest that
each component name includes a numbering as suffix for easy tracking.

Components each may have input and/or output `Data` and/or `Model`. For
details on how to use component, please refer to this
[guide](pipeline_component.md).

An example of initializing a component with specified parameter
values:

``` sourceCode python
hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10,
                       early_stopping_rounds=2, validation_freqs=2)
```

### Input

[Input](pipeline_component.md) encapsulates all input of a component,
including `Data`, `Cache`, and `Model` input. To access `input` of a
component, reference its `input` attribute:

```python
input_all = data_transform_0.input
```

### Output

[Output](pipeline_component.md) encapsulates all output result of a
component, including `Data`, `Cache`, and `Model` output. To access
`Output` from a component, reference its `output` attribute:

``` sourceCode python
output_all = data_transform_0.output
```

### Data

`Data` wraps all data-type input and output of components. FATE Pipeline
includes five types of `data`, each is used for different scenario. For
more information, please refer [here](pipeline_component.md).

### Model

`Model` defines model input and output of components. Similar to `Data`,
the two types of `models` are used for different purposes. For more
information, please refer [here](pipeline_component.md).

### Cache

`Caches` wraps cache input and output of `Intersection` component. Only
`Intersection` component may have `cache` input or output. For more
information, please refer [here](pipeline_component.md).

## Build A Pipeline

Below is a general guide to building a pipeline. Please refer to
[mini demo](../../../examples/pipeline/demo/pipeline-mini-demo.py)
for an explained demo.

Once initialized a pipeline, job participants and initiator should be
specified. Below is an example of initial setup of a pipeline:

```python
pipeline = PipeLine()
pipeline.set_initiator(role='guest', party_id=9999)
pipeline.set_roles(guest=9999, host=10000, arbiter=10000)
```

`Reader` is required to read in data source so that other component(s)
can process data. Define a `Reader` component:

``` sourceCode python
reader_0 = Reader(name="reader_0")
```

In most cases, `DataTransform` follows `Reader` to transform data into
DataInstance format, which can then be used for data engineering and
model training. Some components (such as `Union` and `Intersection`) can
run directly on non-DataInstance tables.

All pipeline components can be configured individually for different
roles by setting `get_party_instance`. For instance, `DataTransform`
component can be configured specifically for guest like this:

```python
data_transform_0 = DataTransform(name="data_transform_0")
guest_component_instance = data_transform_0.get_party_instance(role='guest', party_id=9999)
guest_component_instance.component_param(with_label=True, output_format="dense")
```

To include a component in a pipeline, use `add_component`. To add the
`DataTransform` component to the previously created pipeline, try
this:

```python
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
```

### Build Fate NN Model In Keras Style

In pipeline, you can build NN structures in a Keras style. Take Homo-NN
as an example:

First, import Keras and define your nn structures:

```python
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

layer_0 = Dense(units=6, input_shape=(10,), activation="relu")
layer_1 = Dense(units=1, activation="sigmoid")
```

Then, add nn layers into Homo-NN model like using Sequential class in
Keras:

```python
from pipeline.component.homo_nn import HomoNN

# set parameter
homo_nn_0 = HomoNN(name="homo_nn_0", max_iter=10, batch_size=-1, early_stop={"early_stop": "diff", "eps": 0.0001})
homo_nn_0.add(layer_0)
homo_nn_0.add(layer_1)
```

Set optimizer and compile Homo-NN
model:

```python
homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=0.05), metrics=["Hinge", "accuracy", "AUC"],
                  loss="binary_crossentropy")
```

Add it to
pipeline:

```python
pipeline.add_component(homo_nn, data=Data(train_data=data_transform_0.output.data))
```

## Set job provider 

In version 1.7 and above, user can specify the fate's version to submit the job. If it's not specified, 
default version will be used.  

a. set global version
```python
pipeline.set_global_job_provider("fate@1.7.0")
```

b. component with specified version
```python
homo_nn.provider = "fate@1.7.0"
```

## Init Runtime JobParameters

In version 1.7 and above, user no longer needs to initialize the runtime
environment, like 'work\_mode',

## Run A Pipeline

Having added all components, user needs to first compile pipeline before
running the designed job. After compilation, the pipeline can then be
fit(run train job).

```python
pipeline.compile()
pipeline.fit()
```

## Query on Tasks

FATE Pipeline provides API to query component information, including
data, model, and summary. All query API have matching name to
[FlowPy](flow_sdk.md), while Pipeline retrieves and returns query
result directly to user.

```python
summary = pipeline.get_component("hetero_lr_0").get_summary()
```

## Deploy Components

Once fitting pipeline completes, prediction can be run on new data set.
Before prediction, necessary components need to be first deployed. This
step marks selected components to be used by prediction pipeline.

```python
# deploy select components
pipeline.deploy_component([data_transform_0, hetero_lr_0])
# deploy all components
# note that Reader component cannot be deployed. Always deploy pipeline with Reader by specified component list.
pipeline.deploy_component()
```

## Predict with Pipeline

First, initiate a new pipeline, then specify data source used for
prediction.

```python
predict_pipeline = PipeLine()
predict_pipeline.add_component(reader_0)
predict_pipeline.add_component(pipeline,
                               data=Data(predict_input={pipeline.data_transform_0.input.data: reader_0.output.data}))
```

Prediction can then be initiated on the new pipeline.

```python
predict_pipeline.predict()
```

In addition, since pipeline is modular, user may add new components to
the original pipeline before running
prediction.

```python
predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.hetero_lr_0.output.data))
predict_pipeline.predict()
```

If components are checkpoint saved during training process, user may specify which checkpoint model to be used for prediction.
For an example, please refer [here](../../../examples/pipeline/demo/pipeline-deploy-demo.py).

```python
predict_pipeline.predict(components_checkpoint={"hetero_lr_0": {"step_index": 8}})
```

## Save and Recovery of Pipeline

To save a pipeline, just use **dump** interface.

```python
pipeline.dump("pipeline_saved.pkl")
```

To restore a pipeline, use **load\_model\_from\_file** interface.

```python
from pipeline.backend.pipeline import PineLine
PipeLine.load_model_from_file("pipeline_saved.pkl")
```

## Summary Info of Pipeline

To get the details of a pipeline, use **describe** interface, which
prints the "create time" fit or predict state and the constructed dsl if
exists.

```python
pipeline.describe()
```

## Use Online Inference Service(FATE-Serving) with Pipeline

First, trained pipeline must be deployed before loading and binding
model to online service
[FATE-Serving](https://github.com/FederatedAI/FATE-Serving).

```python
# deploy select components
pipeline.deploy_component([data_transform_0, hetero_lr_0])
# deploy all components
# note that Reader component cannot be deployed. Always deploy pipeline with Reader by specifying component list.
pipeline.deploy_component()
```

Then load model, file path to model storage may be supplied.

```python
pipeline.online.load()
```

Last, bind model to chosen service. Optionally, provide select
FATE-Serving address(es).

```python
# by default, bind model to all FATE-Serving addresses
pipeline.online.bind("service_1")
# bind model to specified FATE-Serving address(es) only
pipeline.online.bind("service_1", "127.0.0.1")
```

## Convert Homo Model to Formats from Other Machine Learning System

To convert a trained homo model into formats of other machine learning
system, use **convert** interface.

```python
pipeline.model_convert.convert()
```

## Upload Data

PipeLine provides functionality to upload local data table. Please refer
to [upload demo](../../../examples/pipeline/demo/pipeline-upload.py)
for a quick example. Note that uploading data can be added all at once,
and the pipeline used to perform upload can be either training or
prediction pipeline (or, a separate pipeline as in the demo).

## Pipeline vs. CLI

In the past versions, user interacts with FATE through command line
interface, often with manually configured conf and dsl json files.
Manual configuration can be tedious and error-prone. FATE Pipeline forms
task configure files automatically at compilation, allowing quick
experiment with task design.
