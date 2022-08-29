# Components

Each `Component` wraps a [FederatedML](../../federatedml_component/README.md)
`Module`. `Modules` implement machine learning algorithms on federated
learning, while `Components` provide convenient interface for easy model
building.

## Interface

### Input

`Input` encapsulates all upstream input to a component in a job
workflow. There are three classes of `input`: `data`, `cache`, and
`model`. Not all components have all three classes of input, and a
component may accept only some types of the class. Note that only
`Intersection` may have `cache` input. For information on each
components' input, check the [list](../../federatedml_component/README.md).

Here is an example to access a component's input:

``` sourceCode python
from pipeline.component import DataTransform
data_transform_0 = DataTransform(name="data_transform_0")
input_all = data_transform_0.input
input_data = data_transform_0.input.data
input_model = data_transform_0.input.model
```

### Output

Same as `Input`, `Output` encapsulates output `data`, `cache`, and
`model` of component in a FATE job. Not all components have all classes
of outputs. Note that only `Intersection` may have `cache` output. For
information on each components' output, check the
[list](../../federatedml_component/README.md).

Here is an example to access a component's output:

``` sourceCode python
from pipeline.component import DataTransform
data_transform_0 = DataTransform(name="data_transform_0")
output_all = data_transform_0.output
output_data = data_transform_0.output.data
output_model = data_transform_0.output.model
```

Meanwhile, to download components' output table or model, please use
[task info](#task-info) interface.

### Data

In most cases, data sets are wrapped into `data` when being passed
between modules. For instance, in the [mini
demo](../../../examples/pipeline/demo/pipeline-mini-demo.py), data output of `data_transform_0` is set
as data input to
`intersection_0`.

``` sourceCode python
pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
```

For data sets used in different modeling stages (e.g., train & validate)
of the same component, additional keywords `train_data` and
`validate_data` are used to distinguish data sets. Also from [mini
demo](../../../examples/pipeline/demo/pipeline-mini-demo.py), result from `intersection_0` and
`intersection_1` are set as train and validate data of hetero logistic
regression,
respectively.

``` sourceCode python
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data,
                                              validate_data=intersection_1.output.data))
```

Another case of using keywords `train_data` and `validate_data` is to
use data output from `DataSplit` module, which always has three data
outputs: `train_data`, `validate_data`, and `test_data`.

``` sourceCode python
pipeline.add_component(hetero_lr_0,
                       data=Data(train_data=hetero_data_split_0.output.data.train_data))
```

A special data type is `predict_input`. `predict_input` is only used for
specifying data input when running prediction task.

Here is an example of running prediction with an upstream model within
the same pipeline:

``` sourceCode python
pipeline.add_component(hetero_lr_1,
                       data=Data(predict_input=hetero_data_split_0.output.data.test_data),
                       model=Model(model=hetero_lr_0))
```

To run prediction with with new data, data source needs to be updated in
prediction job. Below is an example from [mini
demo](../../../examples/pipeline/demo/pipeline-mini-demo.py), where data input of original
<span class="title-ref">data\_transform\_0</span> component is set to be the data
output from <span class="title-ref">reader\_2</span>.

``` sourceCode python
reader_2 = Reader(name="reader_2")
reader_2.get_party_instance(role="guest", party_id=guest).component_param(table=guest_eval_data)
reader_2.get_party_instance(role="host", party_id=host).component_param(table=host_eval_data)
# add data reader onto predict pipeline
predict_pipeline.add_component(reader_2)
predict_pipeline.add_component(pipeline,
                               data=Data(predict_input={pipeline.data_transform_0.input.data: reader_2.output.data}))
```

Below lists all five types of `data` and whether `Input` and `Output`
include
them.

| Data Name      | Input | Output | Use Case                                                           |
| -------------- | ----- | ------ | ------------------------------------------------------------------ |
| data           | Yes   | Yes    | single data input or output                                        |
| train\_data    | Yes   | Yes    | model training; output of `DataSplit` component                    |
| validate\_data | Yes   | Yes    | model training with validate data; output of `DataSplit` component |
| test\_data     | No    | Yes    | output of `DataSplit` component                                    |
| predict\_input | Yes   | No     | model prediction                                                   |

Data

All input and output data of components need to be wrapped into `Data`
objects when being passed between components. For information on valid
data types of each component, check the [list](../../federatedml_component/README.md).
Here is a an example of chaining components with different types of data
input and output:

``` sourceCode python
from pipeline.backend.pipeline import Pipeline
from pipeline.component import DataTransform, Intersection, HeteroDataSplit, HeteroLR
# initialize a pipeline
pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest)
# define all components
data_transform_0 = DataTransform(name="data_transform_0")
data_split = HeteroDataSplit(name="data_split_0")
hetero_lr_0 = HeteroLR(name="hetero_lr_0", max_iter=20)
# chain together all components
pipeline.add_component(reader_0)
pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
pipeline.add_component(hetero_data_split_0, data=Data(data=intersection_0.output.data))
pipeline.add_component(hetero_lr_0, data=Data(train_data=hetero_data_split_0.output.data.train_data,
                                              validate_data=hetero_data_split_0.output.data.test_data))
```

### Model

There are two types of `Model`: `model` and`isometric_model`. When the
current component is of the same class as the previous component, if
receiving `model`, the current component will replicate all model
parameters from the previous component. When a model from previous
component is used as input but the current component is of different
class from the previous component, `isometric_model` is used.

Check below for a case from mini demo, where `model` from `data_transform_0` is
passed to `data_transform_1`.

``` sourceCode python
pipeline.add_component(data_transform_1,
                       data=Data(data=reader_1.output.data),
                       model=Model(data_transform_0.output.model))
```

Here is a case of using `isometric model`. `HeteroFeatureSelection` uses
`isometric_model` from `HeteroFeatureBinning` to select the most
important features.

``` sourceCode python
pipeline.add_component(hetero_feature_selection_0,
                       data=Data(data=intersection_0.output.data),
                       isometric_model=Model(hetero_feature_binning_0.output.model))
```

<div class="warning">

<div class="admonition-title">

Warning

</div>

Please note that when using <span class="title-ref">stepwise</span> or
<span class="title-ref">cross validation</span> method, components do
not have `model` output. For information on valid model types of each
components, check the [list](../../federatedml_component/README.md).

</div>

### Cache

`Cache` is only available for `Intersection` component. Please refer
[here](../../../examples/pipeline/intersect/pipeline-intersect-rsa-cache.py)
for an example of using cache with intersection.

Below code sets cache output from `intersection_0` as cache input of
`intersection_1`.

``` sourceCode python
pipeline.add_component(intersection_1, data=Data(data=data_transform_0.output.data), cache=Cache(intersect_0.output.cache))
```

To load cache from another job, use `CacheLoader` component. In this
[demo](../../../examples/pipeline/intersect/pipeline-intersect-rsa-cache-loader.py),
result from some previous job is loaded into `intersection_0` as cache
input.

``` sourceCode python
pipeline.add_component(cache_loader_0)
pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data), cache=Cache(cache_loader_0.output.cache))
```

### Parameter

Parameters of underlying module can be set for all job participants or
per individual participant.

1.  Parameters for all participants may be specified when defining a
    component:

<!-- end list -->

``` sourceCode python
from pipeline.component import DataTransform
data_transform_0 = DataTransform(name="data_transform_0", input_format="dense", output_format="dense",
                  outlier_replace=False)
```

2.  Parameters can be set for each party individually:

<!-- end list -->

``` sourceCode python
# set guest data_transform_0 component parameters
guest_data_transform_0 = data_transform_0.get_party_instance(role='guest', party_id=9999)
guest_data_transform_0.component_param(with_label=True)
# set host data_transform_0 component parameters
data_transform_0.get_party_instance(role='host', party_id=10000).component_param(with_label=False)
```

### Task Info

Output data and model information of `Components` can be retrieved with
Pipeline task info API. Currently Pipeline support these four types of
query on components:

1.  get\_output\_data: returns downloaded output data; use parameter
    <span class="title-ref">limits</span> to limit output lines
2.  get\_output\_data\_table: returns output data table
    information(including table name and namespace)
3.  get\_model\_param: returns fitted model parameters
4.  get\_summary: returns model summary

To obtain output of a component, the component needs to be first
extracted from pipeline:

``` sourceCode python
print(pipeline.get_component("data_transform_0").get_output_data(limits=10))
```

