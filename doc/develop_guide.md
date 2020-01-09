## Developing guides for develop a runnable algorithm module of FATE

In this document, it describes how to develop an algorithm module, which can be callable under the architecture of FATE.

To develop a module, the following 5 steps are needed.

1. define the python parameter object which will be used in this module.

2. define the setting conf json of the module.

3. define the default runtime conf json of the module.

4. define the transfer_variable json if the module needs federation.

5. define your module which should inherit model_base class.

In the following sections we will describe the 5 steps in detail, with toy_example.

### Step 1. Define the parameter object this module will use.

Parameter object is the only way to pass user-define runtime parameters to the developing module, so every module has it's own parameter object.

In order to define a usable parameter object, three steps will be needed.

a. Open a new python file, rename it as xxx_param.py where xxx stands for your module'name, putting it in folder federatedm/param/.
   The class object defined in xxx_param.py should inherit the BaseParam class that define in federatedml/param/base_param.py

b. __init__ of your parameter class should specify all parameters that the module use.

c. Override the check interface of BaseParam, without which will cause not implemented error. Check method is use to validate the parameter variables.

Take hetero lr's parameter object as example, the python file is federatedml/param/logistic_regression_param.py.

firstly, it inherits BaseParam:

    class LogisticParam(BaseParam):
    
secondly, define all parameter variable in __init__ method:
    
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

    
### Step 2. Define the setting conf of the new module.

The purpose to define a setting conf is that fate_flow module extract this file to get the information of how to start program of the module.

a. Define the setting conf in federatedml/conf/setting_conf/, name it as xxx.json, where xxx is the module you want to develop.
   Please note that xxx.json' name "xxx" is very strict, because when fate_flow dsl parser extract the module "xxx" in job dsl, 
   it just concatenates module's name "xxx" with ".json" and retrieve the setting conf in  federatedml/conf/setting_conf/xxx.json.
   
b. Field Specification of setting conf json.
   module_path: the path prefix of the developing module's program.
   default_runtime_conf: the conf where some default parameter variables define, which will be describe in Step 3.
   param_class: the path to find the param_class define in Step 1, it's a concatenation of path of the parameter python file and parameter object name.
   "role": {
       "guest": the path suffix to start the guest program
       "host":  the path suffix to start the host program
       "arbiter" the path suffix to start the arbiter program
   }. 
        What's more, if this module does not need federation, which means all parties start a same program file, "guest|host|arbiter" is another way to define the role keys.
        
Take hetero-lr to explain too, users can find it in federatedml/conf/setting_conf/HeteroLR.json

    {
        "module_path":  "federatedml/logistic_regression/hetero_logistic_regression",
        "default_runtime_conf": "logistic_regression_param.json",
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
    
Have a look at the above content in HeteroLR.json, HeteroLR is a federation module,
its' guest program is define in federatedml/logistic_regression/hetero_logistic_regression/hetero_lr_guest.py and HeteroLRGuest is the guest class object.
The same rules holds in host and arbiter class too. Fate_flow combine's module_path and role's program to run this module.
"param_class" indicates that the parameter class object of HeteroLR is defined in "federatedml/param/logistic_regression_param.py", and the class name is LogisticParam
.
And default runtime conf is in federatedml/param/logistic_regression_param.py.

### Step 3. Define the default runtime conf of this module (Optional)

Default runtime conf set default values for variables defined in parameter class which will be used in case without user configuration.

It should be put in federatedml/conf/default_runtime_conf(match the setting_conf's "default_runtime_conf" field, it's an optional choice to writing such an json file.

For example, in "federatedml/conf/default_runtime_conf/logistic_regression_param.json", default variables of HeteroLR are writing in it.

    {
    "penalty": "L2",
    "optimizer": "sgd",
    "eps": 1e-5,
    "alpha": 0.01,
    "max_iter": 100,
    "converge_func": "diff",
    "re_encrypt_batches": 2,
    "party_weight": 1,
    "batch_size": 320,
    "learning_rate": 0.01,
    "init_param": {
        "init_method": "random_normal"
    },
    ...
    }
    

### Step 4. Define the transfer variable json of this module and generate transfer variable object. (Optional)

This step is needed only when this module is federated, which means there exist information interaction between different parties.
Note that this json file should be put under the fold federatedml/transfer_variable_conf.
In the json file, first thing you need to do is to define the name of the transfer_variable object, for example, like "HeteroLRTransferVariable".
Secondly, define the transfer_variables. The transfer_variable include three fields: 

a. variable name.
b. src: should be one of "guest", "host", "arbiter", it stands for where interactive information is sending from.
c. "dst": list, should be some combinations of "guest", "host", "arbiter", defines where the interactive information is sending to.

The following is the content of "hetero_lr.json".

    {
      "HeteroLRTransferVariable": {
        "paillier_pubkey": {
          "src": "arbiter",
          "dst": [
            "host",
            "guest"
          ]
        },
        "batch_data_index": {
          "src": "guest",
          "dst": [
            "host"
          ]
        },
        ...
      }
    }
 After finish writing this json file, run the python program of federatedml/util/transfer_variable_generator.py, 
 you will get a transfer_variable python class object, in federatedml/util/transfer_variable/xxx_transfer_variable.py, xxx is the file name of this json file.
 
 
### Step 5. Define your module, it should inherit model_base.

The rule of running of module of fate_flow_client is that:

1. retrieves the setting_conf and find the "module" and "role" fields of setting conf.
2. it initializes the running object of every party.
3. calls the run method of running object.
4. calls the save_data method if needed.
5. class the export_model method if needed.

In this section, we describe how to do 3-5. Many common interfaces are provided in federatedml/model_base.py.

a. override run interface if needed.
    The run function holds the form of following.
    
    def run(self, component_parameters=None, args=None):

Both component_parameters and args are dict objects.
"args" contains input data sets and input models of the module in the form of DTable. The naming of each element is defined in user's dsl config file.
On the other hand, "component_parameters" is the parameter variables of this module which is defined in module parameter class mentioned in step 1. These configured parameters are user-defined or taken from default values setting in default runtime conf.

b. Define your save_data interface so that fate-flow can obtain output data through it when needed.

    def save_data(self):
        return self.data_output

c. Similar with part b, define your export_model interface so that fate-flow can obtain output model when needed.
   The format should be a dict contains both "Meta" and "Param" proto buffer generated.

   Here is an example showing how to export model.

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

## Start a modeling task

After finished developing, here is a simple example for starting a modeling task.

#### 1. Upload data
Before starting a task, you need to load data among all the data-providers. To do that, a load_file config is needed to be prepared.  Then run the following command:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_test/upload_data.json

Note: This step is needed for every data-provide node(i.e. Guest and Host).

#### 2. Start your modeling task
In this step, two config files corresponding to dsl config file and component config file should be prepared. Please make sure the table_name and namespace in the conf file match with upload_data conf. should be Then run the following command:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f submitJob -d dsl_test/test_homolr_job_dsl.json -c dsl_test/${your_component_conf_json}

#### 3. Check log files
Now you can check out the log in the following path: ${your_install_path}/logs/{your jobid}.

For more detail information about dsl configure file and parameter configure files, please check out [example doc here](../examples/federatedml-1.x-examples)
