Developing guides
=================
[`中文`_]

.. _中文: develop_guide_zh.rst

Develop a runnable algorithm module of FATE
-------------------------------------------

In this document, it describes how to develop an algorithm module, which can be callable under the architecture of FATE.

To develop a module, the following 5 steps are needed.

1. define the python parameter object which will be used in this module.

2. define the setting conf json of the module.

3. define the transfer_variable json if the module needs federation.

4. define your module which should inherit model_base class.

5. Define the protobuf file required for model saving.

6. (optional) define Pipeline component for your module.

In the following sections we will describe the 5 steps in detail, with toy_example.

Step 1. Define the parameter object this module will use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameter object is the only way to pass user-define runtime parameters to the developing module, so every module has it's own parameter object. In order to define a usable parameter object, three steps will be needed.

a. Open a new python file, rename it as xxx_param.py where xxx stands for your module'name, putting it in folder python/federatedm/param/.
   The class object defined in xxx_param.py should inherit the BaseParam class that define in python/federatedml/param/base_param.py

b. __init__ of your parameter class should specify all parameters that the module use.

c. Override the check interface of BaseParam, without which will cause not implemented error. Check method is use to validate the parameter variables.

Take hetero lr's parameter object as example, the python file is `python/federatedml/param/logistic_regression_param.py <../python/federatedml/param/logistic_regression_param.py>`_

firstly, it inherits BaseParam:

.. code-block:: python
   
   class LogisticParam(BaseParam):
    
secondly, define all parameter variable in __init__ method:
    
.. code-block:: python

    def __init__(self, penalty='L2',
                 eps=1e-5, alpha=1.0, optimizer='sgd', party_weight=1,
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, converge_func='diff',
                 encrypt_param=EncryptParam(), re_encrypt_batches=2,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 need_run=True, predict_param=PredictParam(), cv_param=CrossValidationParam()):
        super(LogisticParam, self).__init__()
        self.penalty = penalty
        self.eps = eps
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.converge_func = converge_func
        self.encrypt_param = copy.deepcopy(encrypt_param)
        self.re_encrypt_batches = re_encrypt_batches
        self.party_weight = party_weight
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)
        self.need_run = need_run
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)

As the example shown above, the parameter can also be a Param class that inherit the BaseParam. The default setting of this kind of parameter is an instance of this class. Then allocated a deepcopy version of this instance to the class attribution. The deepcopy function is used to avoid same pointer risk during the task running.

Once the class defined properly, a provided parameter parser can parse the value of each attribute recursively.

thirdly, override the check interface:

.. code-block:: python

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

    
Step 2. Define the setting conf of the new module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The purpose to define a setting conf is that fate_flow module extract this file to get the information of how to start program of the module.

a. Define the setting conf in `python/federatedml/conf/setting_conf/`, name it as xxx.json, where xxx is the module you want to develop.
   Please note that xxx.json' name "xxx" is very strict, because when fate_flow dsl parser extract the module "xxx" in job dsl, 
   it just concatenates module's name "xxx" with ".json" and retrieve the setting conf in  `python/federatedml/conf/setting_conf/xxx.json`.
   
b. Field Specification of setting conf json.
   
   :module_path: 
      the path prefix of the developing module's program.

   :param_class:
      the path to find the param_class define in Step 1, it's a concatenation of path of the parameter python file and parameter object name.
   
   :role:
      
      :guest: the path suffix to start the guest program
      :host:  the path suffix to start the host program
      :arbiter: the path suffix to start the arbiter program

      What's more, if this module does not need federation, which means all parties start a same program file, "guest|host|arbiter" is another way to define the role keys.
        

Take hetero-lr as an example, users can find it in `python/federatedml/conf/setting_conf/HeteroLR.json <../python/federatedml/conf/setting_conf/HeteroLR.json>`_

.. code-block:: json
    
    {
        "module_path":  "federatedml/logistic_regression/hetero_logistic_regression",
        "param_class" : "federatedml/param/logistic_regression_param.py/LogisticParam",
        "role":
        {
            "guest":
            {
                "program": "hetero_lr_guest.py/HeteroLRGuest"
            },
            "host":
            {
                "program": "hetero_lr_host.py/HeteroLRHost"
            },
            "arbiter":
            {
                "program": "hetero_lr_arbiter.py/HeteroLRArbiter"
            }
        }
    }
    
Have a look at the above content in HeteroLR.json, HeteroLR is a federation module, its' guest program is define in python/federatedml/logistic_regression/hetero_logistic_regression/hetero_lr_guest.py and HeteroLRGuest is the guest class object. The same rules holds in host and arbiter class too. Fate_flow combine's module_path and role's program to run this module. "param_class" indicates that the parameter class object of HeteroLR is defined in "python/federatedml/param/logistic_regression_param.py", and the class name is LogisticParam.


Step 3. Define the transfer variable json of this module and generate transfer variable object. (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is needed only when this module is federated, which means there exists information interaction between different parties.

.. Note:: 
   
   this json file should be put under the folder `transfer_class <../python/federatedml/transfer_variable/transfer_class>`_

In this python file, you would need to create a "transfer_variable" class and inherit the BaseTransferVariables class. Then, define each transfer variable as its attributes. Here is an example to make it more understandable:

.. code-block:: json

    from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


    # noinspection PyAttributeOutsideInit
    class HeteroBoostingTransferVariable(BaseTransferVariables):
        def __init__(self, flowid=0):
            super().__init__(flowid)
            self.booster_dim = self._create_variable(name='booster_dim', src=['guest'], dst=['host'])
            self.stop_flag = self._create_variable(name='stop_flag', src=['guest'], dst=['host'])
            self.predict_start_round = self._create_variable(name='predict_start_round', src=['guest'], dst=['host'])


:name: a string represents variable name
:src: list, should be some combinations of "guest", "host", "arbiter", it stands for where interactive information is sending from.
:dst: list, should be some combinations of "guest", "host", "arbiter", defines where the interactive information is sending to.

After setting that, the following command would help you create corresponding json setting file in `auth_conf <../python/federatedml/transfer_variable/auth_conf>`_  folder where fate_flow can refer to.

.. code-block:: bash

   python fate_arch/federation/transfer_variable/scripts/generate_auth_conf.py federatedml federatedml/transfer_variable/auth_conf
 
 
Step 4. Define your module, it should inherit model_base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rule of running a module with fate_flow_client is that:

1. retrieves the setting_conf and find the "module" and "role" fields of setting conf.
2. it initializes the running object of every party.
3. calls the fit method of running object.
4. calls the save_data method if needed.
5. calls the export_model method if needed.

In this section, we describe how to do 3-5. Many common interfaces are provided in `python/federatedml/model_base.py <../python/federatedml/model_base.py>`_ .

:Override fit interface if needed:
   The fit function holds the form of following.
   
   .. code-block:: python

      def fit(self, train_data, validate_data):

    Both train_data and validate_data(Optional) are Tables from upstream components(DataIO for example). This is the file where you fit logic of model or feature-engineering components located. When starting a training task, this function will be called by model_base automatically.


:Override predict interface if needed:
   The predict function holds the form of following.

   .. code-block:: python

      def predict(self, data_inst):

    Data_inst is a DTable. Similar to fit function, you can define the prediction procedure in the predict function for different roles. When starting a predict task, this function will be called by model_base automatically. Meanwhile, in training task, this function will also be called to predict train data and validation data (if existed). If you are willing to use evaluation component to evaluate your predict result, it should be designed as the following format:

    - for binary, multi-class classification task and regression task, result header should be: ["label", "predict_result", "predict_score", "predict_detail", "type"]
        * label: Provided label
        * predict_result: Your predict result.
        * predict_score: For binary classification task, it is the score of label "1". For multi-class classification, it is the score of highest label. For regression task, it is your predict result.
        * predict_detail: For classification task, it is the detail scores of each class. For regression task, it is your predict result.
        * type: The source of you input data, eg. train or test. It will be added by model_base automatically.
    - There are two Table return in clustering task.
        The format of first Table: ["cluster_sample_count", "cluster_inner_dist", "inter_cluster_dist"]
        * cluster_sample_count: The sample count of each cluster.
        * cluster_inner_dist: The inner distance of each cluster.
        * inter_cluster_dist: The inter distance between each clusters.
        The format of second Table: ["predicted_cluster_index", "distance"]
        * predicted_cluster_index: Your predict label
        * distance: The distance between each sample to its center point.

:Override transform interface if needed:
   The transform function holds the form of following.

   .. code-block:: python

      def transform(self, data_inst):

   This function is used for feature-engineering components in predict task.

Step 5. Define the protobuf file required for model saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Define your save_data interface:
   so that fate-flow can obtain output data through it when needed.

   .. code-block:: python
      
      def save_data(self):
          return self.data_output

To use the trained model through different platform, FATE use protobuf files to save the parameters and model result of a task. When developing your own module, you are supposed to create two proto files which defined your model content in `this folder <python/federatedml/protobuf/proto>`_.

For more details of protobuf, please refer to `this tutorial <https://developers.google.com/protocol-buffers/docs/pythontutorial>`_

The two proto files are
1. File with "meta" as suffix: Save the parameters of a task.
2. File with "param" as suffix: Save the model result of a task.

After defining your proto files, you can use the following script named `proto_generate.sh <python/federatedml/protobuf/proto_generate.sh>`_ to create the corresponding python file:

   .. code-block:: bash

      bash proto_generate.sh

:Define export_model interface:
   Similar with part b, define your export_model interface so that fate-flow can obtain output model when needed. The format should be a dict contains both "Meta" and "Param" proto buffer generated. Here is an example showing how to export model.

   .. code-block:: python

      def export_model(self):
          meta_obj = self._get_meta()
          param_obj = self._get_param()
          result = {
              self.model_meta_name: meta_obj,
              self.model_param_name: param_obj
          }
          return result


Step 6. Define Pipeline component for your module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One wrapped into a component, module can be used with FATE Pipeline API.
To define a Pipeline component, follow these guidelines:

1. all components reside in ``fate_client/pipeline/component`` directory
2. components should inherit common base ``Component``
3. as a good practice, components should have the same names as their corresponding modules
4. components take in parameters at initialization as defined in pipeline/param, where a BaseParam and consts file are provided
5. set attributes of component input and output,
   including whether module has output model, or type of data output('single' vs. 'multi')

Then you may use Pipeline to construct and initiate a job with the newly defined component.
For guide on Pipeline usage, please refer to `fate_client/pipeline`.

Start a modeling task
---------------------

After finished developing, here is a simple example for starting a modeling task.

:1. Upload data:
   Before starting a task, you need to load data among all the data-providers. To do that, a load_file config is needed to be prepared.  Then run the following command:
   
   .. code-block:: bash
      
      flow data upload -c upload_data.json

   .. Note::
     
      This step is needed for every data-provide node(i.e. Guest and Host).

:2. Start your modeling task:
   In this step, two config files corresponding to dsl config file and component config file should be prepared. Please make sure that the table_name and namespace in the conf file match with upload_data conf. Then run the following command:
  
   .. code-block:: bash

      flow job submit -d ${your_dsl_file.json} -c ${your_component_conf_json}

   If you have defined Pipeline component for your module, you can also make a pipeline script and start your task by:

.. code-block:: bash

      python ${your_pipeline.py}

:3. Check log files:
   Now you can check out the log in the following path: `${your_install_path}/logs/{your jobid}`.

For more detailed information about dsl configure file and parameter configure files, please check out `examples/dsl/v2`.