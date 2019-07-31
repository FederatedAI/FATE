## Developing guides for develop a runnable algorithm module of FATE

In this document, it describes how to develop an algorithm module, which can be callable under the architecture of FATE.

To develop a module, the following 5 steps will be needed.

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
   The class object define it xxx_param.py should inherit the BaseParam class that define in federatedml/param/base_param.py

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

If users want to use toy_example using fate_flow, it should use module "SecureAddExample"

    {
        "module_path":  "federatedml/toy_example",
        "default_runtime_conf": "secure_add_example_param.json",
        "param_class" : "federatedml/param/secure_add_example_param.py/SecureAddExampleParam",
        "role":
        {
            "guest":
            {
                "program": "secure_add_guest.py/SecureAddGuest"
            },
            "host":
            {
                "program": "secure_add_host.py/SecureAddHost"
            }
        }
    }
    
Have a look at the above content in SecureAddExample.json, SecureAddExample is a federation module,
its' guest program is define in federatedml/toy_example/secure_add_guest.py and SecureAddGuest is the guest class object.
The same rules holds in host class too. Fate_flow combine's module_path and role's program to run this module.
"param_class" indicates that the parameter class object of SecureAddExample is define in "federatedml/param/secure_add_example_param.py", and the class name is SecureAddExampleParam.
Besides, default runtime conf is in federatedml/param/secure_add_example_param.py/SecureAddExampleParam.

### Step 3. Define the default runtime conf of this module (Optional)

Default runtime conf set default values for variables defined in parameter class which will be used in case without user configuration.

It should be put in federatedml/conf/default_runtime_conf(match the setting_conf's "default_runtime_conf" field, it's an optional choice to writing such an json file.

For example, in "federatedml/conf/default_runtime_conf/secure_add_example_param.json", default variables of SecureAddExampleParam are writing in it.

    {
        "data_num": 1000,
        "partition": 1,
        "seed": 123
    }
    

### Step 4. Define the transfer variable json of this module and generate transfer variable object. (Optional)

This step is needed only when this module is federated, which means there exist information interaction between different parties.
Note that this json file should be put under the fold federatedml/transfer_variable_conf.
In the json file, first thing you need to do is to define the name of the transfer_variable object, for example, like "SecureAddExampleTransferVariable".
Secondly, define the transfer_variables. The transfer_variable include three fields: 

a. variable name.
b. src: should be one of "guest", "host", "arbiter", it stands for where interactive information is sending from.
c. "dst": list, should be some combinations of "guest", "host", "arbiter", defines where the interactive information is sending to.

The following is the content of "secure_add_example.json".

    {
      "SecureAddExampleTransferVariable": {
        "guest_share": {
          "src": "guest",
          "dst": [
            "host"
          ]
        },
        "host_share": {
          "src": "host",
            "dst": [
              "guest"
           ]
        },
        "host_sum": {
           "src": "host",
           "dst": [
             "guest"
           ]
        }
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

c. Similar with part b, define your export_model interface so that fate-flow can obtain output model when needed.
   The format should be a dict contains both "Meta" and "Param" proto buffer generated.


        
