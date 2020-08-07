# FATE Pipeline

Pipeline is a high-level python API that allows user to design, start, and query FATE jobs in a sequential manner. 
FATE Pipeline is designed to be user-friendly and consistent in behavior with FATE commandline tools. 
User can customize jobs by adding components to pipeline and then initiate the job with one call. 
In addition, Pipeline provides functionality to run prediction and query information after fitting a pipeline.
Run the mini example `pipeline-min-demo` to have a taste of how FATE Pipeline works. 
The default values of party ids and work mode need to be modified for cluster deployment.

```bash
python run_mini_demo.py -mode 0 -guest 9999 -host -10000
```

## A FATE Job is A Sequence

A FATE job includes a sequence of tasks. FATE pipeline provides easy-to-use tools to configure order and setting of the tasks. 

FATE is written in a modular style. Modules are designed to have input and output data and model. 
Therefore two modules are connected when output of one module is set to be input of another module. 
By tracing how one data set is processed through FATE modules in a task, we can see that a FATE job is in fact formed by a sequence of tasks. 
For example, in the mini demo above, guest's data is first read in by `Reader`, then loaded into `DataIO`. 
Overlapping ids between guest and host is then found by running data through `Intersection`. Finally, `HeteroLR` model is run on the data. 
Each of the listed modules run a small task with the data, and together they accomplish a model training job.

Going beyond the given mini demo, a job may include multiple data sets and models. For more pipeline examples, please refer to [demos](./demo/).


## Interface of Pipeline

### Component
FATE modules are each wrapped into `component` in Pipeline API. Each component can take in and output `Data` and `Model`. 
Components can be set

### Data 
In most cases, data set(s) should be set as `data`. 
For data sets used in different stages (e.g., train & validate) within a single component, 
keywords `train_data`, `validate_data`, and `test_data` are used to distinguish data sets.
    
### Model
"Model" defines model input and output of components. There are two types of `Model`: `isometric_model` and `model`.
When the current component is of the same class as the previous component, if receiving `model`,
the current model will replicate all model parameters from the previous model.

When a model from previous component is used but the current component is of different class from the previous component, `isometric_model` is used.
For instance, `HeteroFeatureSelection` uses `isometric_model` from `HeteroFeatureBinning` to select most important features. 

### Output
"Output" encapsulates all output result of a component, including both `Data` and `Model` output.


## Build Pipeline

Below is a general guide to build a pipeline. Please refer to demo `pipeline-mini-demo.py` for an explained demo.

Once initialized a pipeline, job participants and initiator should be specified.

To include a component in a pipeline, a component instance should be first initiated and then added into the pipeline using `add_component`. 
Components 

A pipeline job should start with Reader component. In most cases, DataIO follows Reader to load data into DataInstance format,
which can then be used for data engineering and model training. 
Some components (such as `Union` and `Intersection`) can run directly on non-DataInstance tables.

All pipeline components can be configured individually for different roles by setting `get_party_instance`. 

## Predict with Pipeline

## Query on Tasks

## Deployment 

## Pipeline vs. Commandline 

In the past versions, user interacts with FATE with commandline, often with manual-configured conf and dsl file in json format.
Manual configuration can be tedious and error-prone. Pipeline forms task configure files automatically once 