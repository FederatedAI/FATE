# FATE Pipeline

Pipeline is a high-level python API that allows user to design, start, and query FATE jobs in a sequential manner. 
FATE Pipeline is designed to be user-friendly and consistent in behavior with FATE command line tools. 
User can customize job workflow by adding components to pipeline and then initiate a job with one call. 
In addition, Pipeline provides functionality to run prediction and query information after fitting a pipeline.
Run the [mini demo](./demo/pipeline-mini-demo.py) to have a taste of how FATE Pipeline works.
Default values of party ids and work mode in [config file](./demo/config.yaml) may need to be modified depending on setting.

```bash
python pipeline-mini-demo.py
```

For more pipeline demo, please refer to [demos](./demo).

## A FATE Job is A Sequence

A FATE job includes a sequence of tasks. FATE pipeline provides easy-to-use tools to configure order and setting of the tasks. 

FATE is written in a modular style. Modules are designed to have input and output data and model. 
Therefore two modules are connected when output of one module is set to be input of another module. 
By tracing how one data set is processed through FATE modules in a task, we can see that a FATE job is in fact formed by a sequence of tasks. 
For example, in the [mini demo](./demo/pipeline-mini-demo.py) above, guest's data is first read in by `Reader`, then loaded into `DataIO`. 
Overlapping ids between guest and host are then found by running data through `Intersection`. Finally, `HeteroLR` model is run on the data. 
Each of the listed modules run a small task with the data, and together they constitute a model training job.

Beyond the given mini demo, a job may include multiple data sets and models. For more pipeline examples, please refer to [demos](./demo/).


## Interface of Pipeline

### Component
FATE modules are each wrapped into `component` in Pipeline API. Each component can take in and output `Data` and `Model`. 
Parameters of components can be set conveniently at the time of initialization. Unspecified parameters will take default values. 
All components must have a `name`, whose numbering suffix starts at "0". 
A component's name is its identifier, and so it must be unique within a pipeline.

An example of initializing a component with specified parameter values:
```python
hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10,
                       early_stopping_rounds=2, validation_freqs=2)
```

### Data 
In most cases, data set(s) should be set as `data`. 
For instance, in the mini demo, data output of `dataio_0` is set as data input to `intersection_0`.

```python
pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
```

For data sets used in different stages (e.g., train & validate) within a single component, 
keywords `train_data`, `validate_data`, and `test_data` are used to distinguish data sets.
Also from mini demo, result from `intersection_0` and `intersection_1` are set as train and validate data input to training component, respectively.

```python
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data, validate_data=intersection_1.output.data))
```
    
### Model
`Model` defines model input and output of components. There are two types of `Model`: `model` and`isometric_model`.
When the current component is of the same class as the previous component, if receiving `model`,
the current model will replicate all model parameters from the previous model.

Check below for a case from mini demo, where `model` from `dataio_0` is passed to `dataio_1`.

```python
pipeline.add_component(dataio_1, data=Data(data=input_1.data), model=Model(dataio_0.output.model_output))
```

When a model from previous component is used but the current component is of different class from the previous component, `isometric_model` is used.
For instance, `HeteroFeatureSelection` uses `isometric_model` from `HeteroFeatureBinning` to select most important features. 


### Output
`Output` encapsulates all output result of a component, including `Data` and `Model` output. 
 To access `Output` from a component, reference its `output` attribute:

```python
output_all = dataio_0.output
output_data = dataio_0.output.data
output_model = dataio_0.output.model_output
```

## Build A Pipeline

Below is a general guide to build a pipeline. Please refer to [mini demo](./demo/pipeline-mini-demo.py) for an explained demo.

Once initialized a pipeline, job participants and initiator should be specified.
Below is an example of initial setup of a pipeline:

```python
pipeline = PipeLine()
pipeline.set_initiator(role='guest', party_id=10000)
pipeline.set_roles(guest=10000, host=9999, arbiter=10002)
```

`Reader` is required to read in data source so that other component(s) can process data. 
Define a `Reader` component:

```python
reader_0 = Reader(name="reader_0")
```

In most cases, `DataIO` follows `Reader` to transform data into DataInstance format,
which can then be used for data engineering and model training. 
Some components (such as `Union` and `Intersection`) can run directly on non-DataInstance tables.

All pipeline components can be configured individually for different roles by setting `get_party_instance`. 
For instance, `DataIO` component can be configured specifically for guest like this:

```python
dataio_0 = DataIO(name="dataio_0")
guest_component_instance = dataio_0.get_party_instance(role='guest', party_id=10000)
guest_component_instance.algorithm_param(with_label=True, output_format="dense")
```

To include a component in a pipeline, use `add_component`. 
To add the `DataIO` component to the previously created pipeline, try this:

```python
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
```

## Run A Pipeline

Having added all components, user needs to first compile pipeline before running any job. 
After compilation, the pipeline can then be fit(run train job) with appropriate `Backend` and `WorkMode`.

```python
pipeline.compile()
pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)
```

## Predict with Pipeline

Once pipeline completes fit, prediction can be run on new data set. 
```python
pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)
```

In addition, since pipeline is modular, new components can be added to the original pipeline when running prediction. 

## Query on Tasks

FATE Pipeline also provides API to query component information, including data, model, and metrics.
All query API have matching name to [FlowPy](../fate_flow/doc), while Pipeline retrieves and returns query result directly to user. 

```python
summary = pipeline.get_component("hetero_lr_0").get_summary()
```

## Deployment 

After fitting a pipeline, user may deploy the result model to online service. 

## Pipeline vs. CLI 

In the past versions, user interacts with FATE through command line interface, often with manual-configured conf and dsl json files.
Manual configuration can be tedious and error-prone. Pipeline forms task configure files automatically at compilation. 
