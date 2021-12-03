# Developing guides

[[中文](develop_guide.zh.md)]

## Develop a runnable algorithm component of FATE

In this document, it describes how to develop an algorithm component, which
can be callable under the architecture of FATE.

To develop a component, the following 6 steps are needed.

1.  define the python parameter object which will be used in this
    component.
2.  register meta of the new component.
3.  define the `transfer_variable` object if the component needs federation.
4.  define your component which should inherit `model_base` class.
5.  Define the protobuf file required for model saving.
6.  (optional) define Pipeline component for your component.

In the following sections we will describe the 6 steps in detail, with
`hetero_lr`.

### Step 1. Define the parameter object this component will use

Parameter object is the only way to pass user-define runtime parameters
to the developing component, so every component has it's own parameter object.
In order to define a usable parameter object, three steps will be
needed.

1.  Open a new python file, rename it as `xxx_param.py` where xxx stands
    for your component'name, putting it in folder python/federatedm/param/.
    The class object defined in `xxx_param.py` should inherit the
    BaseParam class that define in
    `python/federatedml/param/base_param.py`
2.  `__init__` of your parameter class should specify all parameters
    that the component use.
3.  Override the check interface of BaseParam, without which will cause
    not implemented error. Check method is use to validate the parameter
    variables.

Take hetero lr's parameter object as example, the python file is
[here](../../python/federatedml/param/logistic_regression_param.py)

#### firstly, it inherits BaseParam:

```python
class LogisticParam(BaseParam):
```

#### secondly, define all parameter variable in `__init__` method:

```python
def __init__(self, penalty='L2',
                 tol=1e-4, alpha=1.0, optimizer='rmsprop',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, early_stop='diff', encrypt_param=EncryptParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 multi_class='ovr', validation_freqs=None, early_stopping_rounds=None,
                 stepwise_param=StepwiseParam(), floating_point_precision=23,
                 metrics=None,
                 use_first_metric_only=False,
                 callback_param=CallbackParam()
                 ):
        super(LogisticParam, self).__init__()
        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.encrypt_param = encrypt_param
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.decay = decay
        self.decay_sqrt = decay_sqrt
        self.multi_class = multi_class
        self.validation_freqs = validation_freqs
        self.stepwise_param = copy.deepcopy(stepwise_param)
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics or []
        self.use_first_metric_only = use_first_metric_only
        self.floating_point_precision = floating_point_precision
        self.callback_param = copy.deepcopy(callback_param)
```

As the example shown above, the parameter can also be a Param class that
inherit the BaseParam. The default setting of this kind of parameter is
an instance of this class. Then allocated a deepcopy version of this
instance to the class attribution. The deepcopy function is used to
avoid same pointer risk during the task running.

Once the class defined properly, a provided parameter parser can parse
the value of each attribute recursively.

#### thirdly, override the check interface:

```python
def check(self):
    descr = "logistic_param's"

    if type(self.penalty).__name__ != "str":
        raise ValueError(
            "logistic_param's penalty {} not supported, should be str type".format(self.penalty))
    else:
        self.penalty = self.penalty.upper()
        if self.penalty not in ['L1', 'L2', 'NONE']:
            raise ValueError(
                "logistic_param's penalty not supported, penalty should be 'L1', 'L2' or 'none'")

    if type(self.eps).__name__ != "float":
        raise ValueError(
            "logistic_param's eps {} not supported, should be float type".format(self.eps))
```

### Step 2. Register meta of the new component 

The purpose to register the meta is that FATE Flow uses 
this file to get the information on  how to start program of the component.

1.  Define component meta python file under
    [components](../../python/federatedml/components), 
    name it as xxx.py, where xxx stands for the algorithm component you want to develop.

2.  Developing the meta file. 
    
      - inherit from ComponentMeta, and name meta with component's name, 
      like xxx_cpn_meta = ComponentMeta("XXX"). XXX is the module to be used in dsl file.  
        
        ```python
          from .components import ComponentMeta
          hetero_lr_cpn_meta = ComponentMeta("HeteroLR")
        ``` 
      - use the decorator `xxx_cpn_meta.bind_runner.on_$role` to bind the running object to each role.  
        $role mainly includes `guest`, `host` and `arbiter`. If component uses the same running module for several roles, syntax like 
        `xxx_cpn_meta.bind_runner.on_$role1.on_$role2.on_$role3` is also supported.   
        This function imports and returns the running object of corresponding role.  
  
        Take hetero-lr as an example, users can find it in
        [python/federatedml/components/hetero_lr.py](../../python/federatedml/components/hetero_lr.py)
      
        ```python
        @hetero_lr_cpn_meta.bind_runner.on_guest
        def hetero_lr_runner_guest():
            from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_guest import HeteroLRGuest
            
            return HeteroLRGuest
            
        @hetero_lr_cpn_meta.bind_runner.on_host
        def hetero_lr_runner_host():
            from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_host import HeteroLRHost
            
            return HeteroLRHost
        ``` 
      - use the decorator `xxx_cpn_meta.bind_param` to bind the parameter object to the developing component, which defines in Step 1.  
        The function imports and returns the parameter object.  
        
        ```python
        @hetero_lr_cpn_meta.bind_param
        def hetero_lr_param():
            from federatedml.param.logistic_regression_param import HeteroLogisticParam
            
            return HeteroLogisticParam
        ``` 
        
### Step 3. Define the transfer variable object of this module. (Optional)

This step is needed only when module is federated, which means
there exists information interaction between different parties.

Developing a file to define transfer_class object under the fold
[`transfer_class`](../../python/federatedml/transfer_variable/transfer_class)

In this python file, you would need to create a `transfer_variable`
class which inherits `BaseTransferVariables`. Then, define each
transfer variable as its attributes. Here is an example:

```python
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables

# noinspection PyAttributeOutsideInit
class HeteroLRTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.batch_data_index = self._create_variable(name='batch_data_index', src=['guest'], dst=['host'])
        self.batch_info = self._create_variable(name='batch_info', src=['guest'], dst=['host', 'arbiter'])
        self.converge_flag = self._create_variable(name='converge_flag', src=['arbiter'], dst=['host', 'guest'])
        self.fore_gradient = self._create_variable(name='fore_gradient', src=['guest'], dst=['host'])
        self.forward_hess = self._create_variable(name='forward_hess', src=['guest'], dst=['host'])
        self.guest_gradient = self._create_variable(name='guest_gradient', src=['guest'], dst=['arbiter'])
        self.guest_hess_vector = self._create_variable(name='guest_hess_vector', src=['guest'], dst=['arbiter'])
        self.guest_optim_gradient = self._create_variable(name='guest_optim_gradient', src=['arbiter'], dst=['guest'])
        self.host_forward_dict = self._create_variable(name='host_forward_dict', src=['host'], dst=['guest'])
        self.host_gradient = self._create_variable(name='host_gradient', src=['host'], dst=['arbiter'])
        self.host_hess_vector = self._create_variable(name='host_hess_vector', src=['host'], dst=['arbiter'])
        self.host_loss_regular = self._create_variable(name='host_loss_regular', src=['host'], dst=['guest'])
        self.host_optim_gradient = self._create_variable(name='host_optim_gradient', src=['arbiter'], dst=['host'])
        self.host_prob = self._create_variable(name='host_prob', src=['host'], dst=['guest'])
        self.host_sqn_forwards = self._create_variable(name='host_sqn_forwards', src=['host'], dst=['guest'])
        self.loss = self._create_variable(name='loss', src=['guest'], dst=['arbiter'])
        self.loss_intermediate = self._create_variable(name='loss_intermediate', src=['host'], dst=['guest'])
        self.paillier_pubkey = self._create_variable(name='paillier_pubkey', src=['arbiter'], dst=['host', 'guest'])
        self.sqn_sample_index = self._create_variable(name='sqn_sample_index', src=['guest'], dst=['host'])
        self.use_async = self._create_variable(name='use_async', src=['guest'], dst=['host'])
```

  - name  
    a string represents variable name

  - src  
    list, should be some combinations of `guest`, `host`, `arbiter`, it
    stands for where interactive information is sending from.

  - dst  
    list, should be some combinations of `guest`, `host`, `arbiter`,
    defines where the interactive information is sending to.


### Step 4. Define your module, it should inherit `model_base`

The rule of running a module with `fate_flow_client` is that:

1.  retrieves component registration from database and find the running object of each role.
2.  it initializes the running object of every party.
3.  calls the fit method of running object.
4.  calls the `save_data` method if needed.
5.  calls the `export_model` method if needed.

In this section, we describe how to do 2-5. Many common interfaces are
provided in
[python/federatedml/model\_base.py](../../python/federatedml/model_base.py)
 
  - Override `__init__` interface    
    Specify the class of model parameter which is already defined in [Step1](#step-1-define-the-parameter-object-this-component-will-use).  
    Take `hetero_lr_base.py` as an example, the last line specify the parameter class of the your model.
    
      ```python
        def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.mode = consts.HETERO
        self.aggregator = None
        self.cipher = None
        self.batch_generator = None
        self.gradient_loss_operator = None
        self.converge_procedure = None
        self.model_param = HeteroLogisticParam()
      ```
    Note: This step is mandatory. If you do not assign the vale of `self.model_param`, you will not be able to access the value of the model parameter in function `_init_model(self, params)`. 
  - Override fit interface if needed  
    The fit function holds the form of following.
    
    ```python
    def fit(self, train_data, validate_data):
    ```
    
    Both `train_data` and `validate_data`(Optional) are Tables from 
    upstream components(DataIO for example). This is the file where
    you fit logic of model or feature-engineering components located.
    When starting a training task, this function will be called by
    `model_base` automatically.

  - Override predict interface if needed  
    The predict function holds the form of following.
    
    ```python
    def predict(self, data_inst):
    ```
    
    `data_inst` is a DTable. Similar to fit function, you can define
    the prediction procedure in the predict function for different
    roles. When starting a predict task, this function will be called
    by `model_base` automatically. Meanwhile, in training task, this
    function will also be called to predict train data and validation
    data (if existed). If you are willing to use evaluation component
    to evaluate your predict result, it should be designed as the
    following format:
    
    - for binary, multi-class classification task and regression task, result header should be: ["label", "predict_result", "predict_score", "predict_detail", "type"]
       
      - `label`: Provided label
      - `predict_result`: Your predict result.
      - `predict_score`: For binary classification task, it is the score of label "1".
        For multi-class classification, it is the score of highest label.
        For regression task, it is your predict result.
      - `predict_detail`: For classification task, it is the detail scores of each class
        For regression task, it is your predict result.
      - `type`: The source of you input data, eg. train or test.
        It will be added by `model_base` automatically.

  - There are two Table return in clustering task.  
    
    The format of first Table: ["cluster_sample_count", "cluster_inner_dist", "inter_cluster_dist"]
      
      - `cluster_sample_count`: The sample count of each cluster.
      - `cluster_inner_dist`: The inner distance of each cluster.
      - `inter_cluster_dist`: The inter distance between each clusters.
      
    The format of second Table:["predicted_cluster_index", "distance"]
      
      - `predicted_cluster_index`: Your predict label
      - `distance`: The distance between each sample to its center point.

  - Override transform interface if needed  
    The transform function holds the form of following.
    
    ```python
    def transform(self, data_inst):
    ```
    
    This function is used for feature-engineering components in predict
    task.

  - Define your `save_data` interface  
    so that fate-flow can obtain output data through it when needed.
    
    ```python
    def save_data(self):
        return self.data_output
    ```

### Step 5. Define the protobuf file required for model saving

#### define proto buffer
To use the trained model through different platform, FATE use protobuf
files to save the parameters and model result of a task. When developing
your own module, you are supposed to create two proto files which
defined your model content in [this folder](../../python/federatedml/protobuf/proto).

For more details of protobuf, please refer to [this
tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial)

The two proto files are 

1. File with "meta" as suffix: Save the parameters of a task.
2. File with "param" as suffix: Save the model result of a task.

After defining your proto files, you can use the following script named
[generate\_py.sh](../../python/fate_arch/protobuf/generate_py.sh) to create
the corresponding python file:

 
```bash
bash generate_py.sh
```

#### Define `export_model` interface  

Similar with part b, define your `export_model` interface so that
fate-flow can obtain output model when needed. The format should be
a dict contains both "Meta" and "Param" proto buffer generated. Here
is an example showing how to export model.

```python
def export_model(self):
    meta_obj = self._get_meta()
    param_obj = self._get_param()
    result = {
        self.model_meta_name: meta_obj,
        self.model_param_name: param_obj
    }
    return result
```

### Step 6. Define Pipeline component for your module

One wrapped into a component, module can be used with FATE Pipeline API.
To define a Pipeline component, follow these guidelines:

1.  all components reside in
    [fate_client/pipeline/component](../../python/fate_client/pipeline/component)
    directory
2.  components should inherit common base `Component`
3.  as a good practice, components should have the same names as their
    corresponding modules
4.  components take in parameters at initialization as defined in
    [fate_client/pipeline/param](../../python/fate_client/pipeline/param),
    where a BaseParam and consts file are provided
5.  set attributes of component input and output, including whether
    module has output model, or type of data output('single' vs.
    'multi')

Then you may use Pipeline to construct and initiate a job with the newly
defined component. For guide on Pipeline usage, please refer to
[fate_client/pipeline](../api/fate_client/pipeline.md).

## Start a modeling task

After finished developing, here is a simple example for starting a
modeling task.

### 1. Upload data  

Before starting a task, you need to load data among all the
data-providers. To do that, a `load_file` config is needed to be
prepared. Then run the following command:
    
```bash
flow data upload -c upload_data.json
```
    
!!!Note
    
    This step is needed for every data-provide node(i.e. Guest and Host).
    

### 2. Start your modeling task  

In this step, two config files corresponding to dsl config file and
component config file should be prepared. Please make sure that the
`table_name` and `namespace` in the conf file match with `upload_data`
conf. Then run the following
command:

```bash
flow job submit -d ${your_dsl_file.json} -c ${your_component_conf_json}
```

If you have defined Pipeline component for your module, you can also
make a pipeline script and start your task by:


```bash
python ${your_pipeline.py}
```

### 3. Check log files  

Now you can check out the log in the path: `$PROJECT_BASE/logs/${your jobid}`

For more detailed information about dsl configure file and parameter
configure files, please check out `examples/dsl/v2`
