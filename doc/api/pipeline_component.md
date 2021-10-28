Components
==========

Each `Component` wraps a [FederatedML](./federatedml_module.md)
`Module`. `Modules` implement machine learning algorithms on federated
learning, while `Components` provide convenient interface for easy model
building.

Interface
---------

### Input

`Input` encapsulates all upstream input to a component in a job
workflow. There are three classes of `input`: `data`, `cache`, and
`model`. Not all components have all three classes of input, and a
component may accept only some types of the class. Note that only
`Intersection` may have `cache` input. For information on each
components\' input, check the [list](#component-list) below.

Here is an example to access a component\'s input:

``` {.sourceCode .python}
from pipeline.component import DataIO
dataio_0 = DataIO(name="dataio_0")
input_all = dataio_0.input
input_data = dataio_0.input.data
input_model = dataio_0.input.model
```

### Output

Same as `Input`, `Output` encapsulates output `data`, `cache`, and
`model` of component in a FATE job. Not all components have all classes
of outputs. Note that only `Intersection` may have `cache` output. For
information on each components\' output, check the
[list](#component-list) below.

Here is an example to access a component\'s output:

``` {.sourceCode .python}
from pipeline.component import DataIO
dataio_0 = DataIO(name="dataio_0")
output_all = dataio_0.output
output_data = dataio_0.output.data
output_model = dataio_0.output.model
```

Meanwhile, to download components\' output table or model, please use
[task info](#task-info) interface.

### Data

In most cases, data sets are wrapped into `data` when being passed
between modules. For instance, in the [mini
demo](../demo/pipeline-mini-demo.py), data output of `dataio_0` is set
as data input to `intersection_0`.

``` {.sourceCode .python}
pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
```

For data sets used in different modeling stages (e.g., train & validate)
of the same component, additional keywords `train_data` and
`validate_data` are used to distinguish data sets. Also from [mini
demo](../demo/pipeline-mini-demo.py), result from `intersection_0` and
`intersection_1` are set as train and validate data of hetero logistic
regression, respectively.

``` {.sourceCode .python}
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data,
                                              validate_data=intersection_1.output.data))
```

Another case of using keywords `train_data` and `validate_data` is to
use data output from `DataSplit` module, which always has three data
outputs: `train_data`, `validate_data`, and `test_data`.

``` {.sourceCode .python}
pipeline.add_component(hetero_lr_0,
                       data=Data(train_data=hetero_data_split_0.output.data.train_data))
```

A special data type is `predict_input`. `predict_input` is only used for
specifying data input when running prediction task.

Here is an example of running prediction with an upstream model within
the same pipeline:

``` {.sourceCode .python}
pipeline.add_component(hetero_lr_1,
                       data=Data(predict_input=hetero_data_split_0.output.data.test_data),
                       model=Model(model=hetero_lr_0))
```

To run prediction with with new data, data source needs to be updated in
prediction job. Below is an example from [mini
demo](../demo/pipeline-mini-demo.py), where data input of original
[dataio\_0]{.title-ref} component is set to be the data output from
[reader\_2]{.title-ref}.

``` {.sourceCode .python}
reader_2 = Reader(name="reader_2")
reader_2.get_party_instance(role="guest", party_id=guest).component_param(table=guest_eval_data)
reader_2.get_party_instance(role="host", party_id=host).component_param(table=host_eval_data)
# add data reader onto predict pipeline
predict_pipeline.add_component(reader_2)
predict_pipeline.add_component(pipeline,
                               data=Data(predict_input={pipeline.dataio_0.input.data: reader_2.output.data}))
```

Below lists all five types of `data` and whether `Input` and `Output`
include them.

  -------------------------------------------------------------------------
  Data Name             Input          Output         Use Case
  --------------------- -------------- -------------- ---------------------
  data                  Yes            Yes            single data input or
                                                      output

  train\_data           Yes            Yes            model training;
                                                      output of `DataSplit`
                                                      component

  validate\_data        Yes            Yes            model training with
                                                      validate data; output
                                                      of `DataSplit`
                                                      component

  test\_data            No             Yes            output of `DataSplit`
                                                      component

  predict\_input        Yes            No             model prediction
  -------------------------------------------------------------------------

  : Data

All input and output data of components need to be wrapped into `Data`
objects when being passed between components. For information on valid
data types of each component, check the [list](#component-list) below.
Here is a an example of chaining components with different types of data
input and output:

``` {.sourceCode .python}
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
```

### Model

There are two types of `Model`: `model` and`isometric_model`. When the
current component is of the same class as the previous component, if
receiving `model`, the current component will replicate all model
parameters from the previous component. When a model from previous
component is used as input but the current component is of different
class from the previous component, `isometric_model` is used.

Check below for a case from mini demo, where `model` from `dataio_0` is
passed to `dataio_1`.

``` {.sourceCode .python}
pipeline.add_component(dataio_1,
                       data=Data(data=reader_1.output.data),
                       model=Model(dataio_0.output.model))
```

Here is a case of using `isometric model`. `HeteroFeatureSelection` uses
`isometric_model` from `HeteroFeatureBinning` to select the most
important features.

``` {.sourceCode .python}
pipeline.add_component(hetero_feature_selection_0,
                       data=Data(data=intersection_0.output.data),
                       isometric_model=Model(hetero_feature_binning_0.output.model))
```

::: {.warning}
::: {.admonition-title}
Warning
:::

Please note that when using [stepwise]{.title-ref} or [cross
validation]{.title-ref} method, components do not have `model` output.
For information on valid model types of each components, check the
[list](#component-list) below.
:::

### Cache

`Cache` is only available for `Intersection` component. Please refer
[here](../../../examples/pipeline/intersect/pipeline-intersect-rsa-cache.py)
for an example of using cache with intersection.

Below code sets cache output from `intersection_0` as cache input of
`intersection_1`.

``` {.sourceCode .python}
pipeline.add_component(intersection_1, data=Data(data=dataio_0.output.data), cache=Cache(intersect_0.output.cache))
```

To load cache from another job, use `CacheLoader` component. In this
[demo](../../../../examples/pipeline/intersect/pipeline-intersect-rsa-cache-loader.py),
result from some previous job is loaded into `intersection_0` as cache
input.

``` {.sourceCode .python}
pipeline.add_component(cache_loader_0)
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data), cache=Cache(cache_loader_0.output.cache))
```

### Parameter

Parameters of underlying module can be set for all job participants or
per individual participant.

1.  Parameters for all participants may be specified when defining a
    component:

``` {.sourceCode .python}
from pipeline.component import DataIO
dataio_0 = DataIO(name="dataio_0", input_format="dense", output_format="dense",
                  outlier_replace=False)
```

2.  Parameters can be set for each party individually:

``` {.sourceCode .python}
# set guest dataio_0 component parameters
guest_dataio_0 = dataio_0.get_party_instance(role='guest', party_id=9999)
guest_dataio_0.component_param(with_label=True)
# set host dataio_0 component parameters
dataio_0.get_party_instance(role='host', party_id=10000).component_param(with_label=False)
```

### Task Info

Output data and model information of `Components` can be retrieved with
Pipeline task info API. Currently Pipeline support these four types of
query on components:

1.  get\_output\_data: returns downloaded output data; use parameter
    [limits]{.title-ref} to limit output lines
2.  get\_output\_data\_table: returns output data table
    information(including table name and namespace)
3.  get\_model\_param: returns fitted model parameters
4.  get\_summary: returns model summary

To obtain output of a component, the component needs to be first
extracted from pipeline:

``` {.sourceCode .python}
print(pipeline.get_component("dataio_0").get_output_data(limits=10))
```

Component List
--------------

Below lists input and output elements of each component.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Algorithm                                                                           Component Name           Description                  Acceptable Input  Acceptable       Acceptable   Acceptable
                                                                                                                                            Data              Output Data      Input Model  Output Model
  ----------------------------------------------------------------------------------- ------------------------ ---------------------------- ----------------- ---------------- ------------ ------------
  Reader                                                                              Reader                   This component is always the None              data             None         None
                                                                                                               first component of a                                                         
                                                                                                               pipeline task(except for                                                     
                                                                                                               upload). It loads raw data                                                   
                                                                                                               from storage.                                                                

  [DataIO](./federatedml/util.md)                                      DataIO                   This component usually       data              data             model        model
                                                                                                               follows `Reader`. It                                                         
                                                                                                               transforms user-uploaded                                                     
                                                                                                               date into Instance                                                           
                                                                                                               object(deprecate in                                                          
                                                                                                               FATe-v1.7, use DataTransform                                                 
                                                                                                               instead).                                                                    

  [DataTransform](./federatedml/util.md)                               DataTransform            This component usually       data              data             model        model
                                                                                                               follows `Reader`. It                                                         
                                                                                                               transforms user-uploaded                                                     
                                                                                                               date into Instance object.                                                   

  [Intersect](./federatedml/statistic/intersect.md)                    Intersection             Compute intersect data set   data              data             None         None
                                                                                                               of multiple parties without                                                  
                                                                                                               leakage of difference set                                                    
                                                                                                               information. Mainly used in                                                  
                                                                                                               hetero scenario task.                                                        

  [Federated Sampling](./federatedml/feature.md)                       FederatedSample          Federated Sampling data so   data              data             model        model
                                                                                                               that its distribution become                                                 
                                                                                                               balance in each party.This                                                   
                                                                                                               module supports standalone                                                   
                                                                                                               and federated versions.                                                      

  [Feature Scale](./federatedml/feature.md)                            FeatureScale             Feature scaling and          data              data             model        model
                                                                                                               standardization.                                                             

  [Hetero Feature Binning](./federatedml/feature.md)                   Hetero Feature Binning   With binning input data,     data              data             model        model
                                                                                                               calculates each column\'s iv                                                 
                                                                                                               and woe and transform data                                                   
                                                                                                               according to the binned                                                      
                                                                                                               information.                                                                 

  [Homo Feature Binning](./federatedml/feature.md)                     Homo Feature Binning     Calculate quantile binning   data              data             None         model
                                                                                                               through multiple parties                                                     

  [OneHot Encoder](./federatedml/feature.md)                           OneHotEncoder            Transfer a column into       data              data             model        model
                                                                                                               one-hot format.                                                              

  [Hetero Feature Selection](./federatedml/feature.md)                 HeteroFeatureSelection   Provide 5 types of filters.  data              data             model;       model
                                                                                                               Each filters can select                                         isometric    
                                                                                                               columns according to user                                       model        
                                                                                                               config                                                                       

  [Union](./federatedml/union.md)                            Union                    Combine multiple data tables List\[data\]      data             model        model
                                                                                                               into one.                                                                    

  [Hetero-LR](./federatedml/logistic_regression.md)       HeteroLR                 Build hetero logistic        train\_data;      data             model        model
                                                                                                               regression module through    validate\_data;                                 
                                                                                                               multiple parties.            predict\_input                                  

  [Local Baseline](./federatedml/local_baseline.md)                    LocalBaseline            Wrapper that runs            train\_data;      data             None         None
                                                                                                               sklearn(scikit-learn)        validate\_data;                                 
                                                                                                               Logistic Regression model    predict\_input                                  
                                                                                                               with local data.                                                             

  [Hetero-LinR](./federatedml/linear_regression.md)       HeteroLinR               Build hetero linear          train\_data;      data             model        model
                                                                                                               regression module through    validate\_data;                                 
                                                                                                               multiple parties.            predict\_input                                  

  [Hetero-Poisson](./federatedml/poisson_regression.md)   HeteroPoisson            Build hetero poisson         train\_data;      data             model        model
                                                                                                               regression module through    validate\_data;                                 
                                                                                                               multiple parties.            predict\_input                                  

  [Homo-LR](./federatedml/logistic_regression.md)         HomoLR                   Build homo logistic          train\_data;      data             model        model
                                                                                                               regression module through    validate\_data;                                 
                                                                                                               multiple parties.            predict\_input                                  

  [Homo-NN](./federatedml/nn/homo_nn.md)                               HomoNN                   Build homo neural network    train\_data;      data             model        model
                                                                                                               module through multiple      validate\_data;                                 
                                                                                                               parties.                     predict\_input                                  

  [Hetero Secure Boosting](./federatedml/ensemble.md)                  HeteroSecureBoost        Build hetero secure boosting train\_data;      data             model        model
                                                                                                               module through multiple      validate\_data;                                 
                                                                                                               parties                      predict\_input                                  

  [Hetero Fast Secure Boosting](./federatedml/ensemble.md)             HeteroFastSecureBoost    Build hetero secure boosting train\_data;      data             model        model
                                                                                                               model through multiple       validate\_data;                                 
                                                                                                               parties in layered/mix       predict\_input                                  
                                                                                                               manners.                                                                     

  [Hetero Secure Boost Feature                                                        SBT Feature Transformer  This component can encode    data              data             model        model
  Transformer](./federatedml/feature.md#sbt_feature_transformer)                                           sample using Hetero SBT leaf                                                 
                                                                                                               indices.                                                                     

  [Evaluation](./federatedml/evaluation.md)                            Evaluation               Output the model evaluation  data;             None             model        None
                                                                                                               metrics for user.            List\[data\]                                    

  [Hetero Pearson](./federatedml/correlation.md)             HeteroPearson            Calculate hetero correlation data              None             model        model
                                                                                                               of features from different                                                   
                                                                                                               parties.                                                                     

  [Hetero-NN](./federatedml/hetero_nn.md)                           HeteroNN                 Build hetero neural network  train\_data;      data             model        model
                                                                                                               module.                      validate\_data;                                 
                                                                                                                                            predict\_input                                  

  [Homo Secure Boosting](./federatedml/ensemble.md)                    HomoSecureBoost          Build homo secure boosting   train\_data;      data             model        model
                                                                                                               module through multiple      validate\_data;                                 
                                                                                                               parties                      predict\_input                                  

  [Homo OneHot Encoder](./federatedml/feature.md)                      HomoOneHotEncoder        Build homo onehot encoder    data              data             model        model
                                                                                                               module through multiple                                                      
                                                                                                               parties.                                                                     

  [Data Split](./federatedml/data_split.md)            Data Split               Split one data table into 3  data              train\_data &    None         None
                                                                                                               tables by given ratio or                       validate\_data &              
                                                                                                               count                                          test\_data                    

  [Column Expand](./federatedml/feature.md)                            Column Expand            Add arbitrary number of      data              data             model        model
                                                                                                               columns with user-provided                                                   
                                                                                                               values.                                                                      

  [Hetero KMeans](./federatedml/hetero_kmeans.md)       Hetero KMeans            Build Hetero KMeans module   train\_data;      data             model        model
                                                                                                               through multiple parties     validate\_data;                                 
                                                                                                                                            predict\_input                                  

  [Data Statistics](./federatedml/statistic.md)                        Data Statistics          Compute Data Statistics      data              data             None         model

  [Scorecard](./federatedml/scorecard.md)                    Scorecard                Scale predict score to       data              data             None         None
                                                                                                               credit score by given                                                        
                                                                                                               scaling parameters                                                           

  [Feldman Verifiable                                                                 Feldman Verifiable Sum   This component will sum      data              data             None         None
  Sum](./federatedml/statistic/feldman_verifiable_sum.md)                                       multiple privacy values                                                      
                                                                                                               without exposing data                                                        

  [Secure Information Retrieval](../../../federatedml/secure_information_retrieval)   Secure Information       This component will securely data              data             None         model
                                                                                      Retrieval                retrieve designated                                                          
                                                                                                               feature(s) or label value                                                    

  [Sample Weight](./federatedml/util.md)                               Sample Weight            This component assigns       data              data             model        model
                                                                                                               weight to instances                                                          
                                                                                                               according to user-specified                                                  
                                                                                                               parameters                                                                   

  [Feature Imputation](./federatedml/feature.md)                       Feature Imputation       This component imputes       data              data             model        model
                                                                                                               missing features using                                                       
                                                                                                               arbitrary methods/values                                                     

  [Label Transform](:%20./federatedml/util.md)                         Label Transform          This component replaces      data              data             model        model
                                                                                                               label with designated values                                                 
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  : Component

Params
------

::: {.automodule}
pipeline/param
:::
