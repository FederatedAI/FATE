## 开发指南：为 FATE 开发可运行的算法模块

本文档描述了如何开发算法模块，使得该模块可以在 FATE 架构下被调用。

要开发模块，需要执行以下 5 个步骤。

1.定义将在此模块中使用的 python 参数对象。

2.定义模块的 json 配置文件。

3.定义模块的默认运行时（runtime）配置文件。

4.如果模块需要联邦，则需定义 transfer_variable.json 文件。

5.定义您的模块继承的 model_base 类。

在以下各节中，我们将通过 toy_example 详细描述这 5 个步骤。

### 第 1 步：定义此模块将使用的参数对象。

参数对象是将用户定义的运行时参数传递给开发模块的唯一方法，因此每个模块都有其自己的参数对象。

为定义可用的参数对象，需要三个步骤。

a. 打开一个新的 python 文件，将其重命名为 xxx_param.py，其中xxx代表您模块的名称，并将其放置在 federatedml/param/ 文件夹中。
在 xxx_param.py 中定义它的类对象，应该继承 federatedml/param/base_param.py 中定义的 BaseParam 类。

b. 参数类的 `__init__` 方法应该指定模块使用的所有参数。

c. 重载 BaseParam 的参数检查接口，否则将会抛出未实现的错误。检查方法被用于验证参数变量是否可用。

以 hetero lr 的参数对象为例，python文件为federatedml/param/logistic_regression_param.py。

首先，它继承自 BaseParam：
```
class LogisticParam(BaseParam):
```
然后，在 `__init__` 方法中定义所有参数变量：
```    
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
```
如上面的示例所示，该参数也可以是 Param 类。此类参数的默认设置是此类的一个实例。然后将该实例的深度复制（`deepcopy`）版本分配给类归属。深度复制功能用于避免任务同时运行时指向相同内存的风险。

一旦正确定义了类，已有的参数解析器就可以递归地解析每个属性的值。

之后，重载参数检查的接口：
```
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
### 第二步：定义新模块的配置文件。

定义配置文件是为了使 `fate_flow` 模块通过该文件以获取有关如何启动模块程序的信息。

a. 在`federatedml/conf/setting_conf/`中定义名为 xxx.json 的配置文件，其中 xxx 是您要开发的模块。请注意，xxx.json 的名称 “xxx” 要求非常严格，因为当 fate_flow dsl 解析器在作业 dsl 中提取模块 “xxx” 时，它只是将模块名称 “xxx” 与 “.json” 连接起来，并在`federatedml/conf/setting_conf/xxx.json`中检索配置文件。

b. 设置 conf.json 的字段规范。
module_path：您开发的模块的路径前缀。
default_runtime_conf：参数变量的缺省配置文件，将在本文第 3 步中详细描述。
param_class：在步骤 1 中定义的 param_class 的路径，它是参数 python 文件和参数对象名称的路径的连结。
```
    "role": {
        "guest": 启动客户机程序的路径后缀
        "host":  启动主机程序的路径后缀
        "arbiter": 启动仲裁程序的路径后缀
    }
```
另外，如果该模块不需要联邦，即各方都可以启动同一个程序文件，那么 `"guest | host | arbiter"` 可以作为定义角色密钥的另一种方法。

也可以用 hetero-lr 来说明，您可以在 `federatedml/conf/setting_conf/HeteroLR.json` 中找到它。
```
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
```
我们来看一下在 HeteroLR.json 里上面这部分内容：HeteroLR 是一个联邦模块，它的Guest程序在 `federatedml/logistic_regression/hetero_logistic_regression/hetero_lr_guest.py` 中定义，并且 HeteroLRGuest 是一个Guest类对象，对于Host和Arbiter类对象也有类似的定义。fate_flow 会结合 module_path 和角色程序来运行该模块。"param_class" 指在 `federatedml/param/logistic_regression_param.py` 中定义了 HeteroLR 的参数类对象，并且类名称为 LogisticParam。默认的运行时配置文件位于 `federatedml/param/logistic_regression_param.py` 中。

### 第三步：定义此模块的默认运行时配置（可选）

缺省运行时配置为参数类中定义的变量设置缺省值。若用户没有配置这些参数，则将使用这些缺省值。应将其放在`federatedml/conf/default_runtime_conf`（与 setting_conf 的 "default_runtime_conf" 字段匹配）。这是编写这些 json 文件时可选项。

例如，HeteroLR 的缺省变量在`federatedml/conf/default_runtime_conf/logistic_regression_param.json`中给出。
```
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
```

### 第四步：定义此模块的传递变量 json 文件并生成传递变量对象。（可选）

仅在此模块被联邦时（即不同参与方之间存在信息交互）才需要执行此步骤。请注意，应将其放在 "federatedml/transfer_variable_conf" 文件夹中。在 json 文件中，您需要做的第一件事就是定义 transfer_variable 对象的名称，例如 “HeteroLRTransferVariable”。然后，定义 transfer_variables。transfer_variable 包含三个字段：

a. 变量名
b. src：应为 "guest"，"host"，"arbiter" 之一，它表示发送交互信息从何处发出。
C. dst：应为 "guest"，"host"，"arbiter" 的某些组合列表，用于定义将交互信息发送到何处。

以下是 “hetero_lr.json” 的内容。
```
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
```
在 json 文件编写完成后，运行 federatedml/util/transfer_variable_generator.py 程序，
您将在 federatedml/util/transfer_variable/xxx_transfer_variable.py 中获得一个 transfer_variable python 类对象，xxx 是此 json 文件的文件名。


### 第五步：定义您的模块（应继承 model_base）。

fate_flow_client 模块的运行规则是：

1.检索 setting_conf 并找到配置文件的“module”和“role”字段。
2.初始化各方的运行对象。
3.调用运行对象的 run 方法。
4.如果需要，调用 save_data 方法。
5.如果需要，调用 export_model 方法。

在本节中，我们讲解如何执行规则 3 至 5 。federatedml/model_base.py 中提供了许多常用接口。

a. 在需要时重载 run 接口。run 功能具有以下形式。
```
    def run(self, component_parameters=None, args=None):
```
component_parameters 和 args 都是 dict 对象。“args”包含 DTable 形式的模块输入数据集和输入模型。每个元素的命名都在用户的 dsl 配置文件中定义。另一方面，“component_parameters” 是此模块的参数变量，该变量在步骤 1 中提到的模块参数类中定义。这些配置的参数是用户定义的，或取自配置文件中的默认值设置。

b. 定义您的 save_data 接口，以便 fate-flow 可以在需要时通过它获取输出数据。
```
    def save_data(self):
        return self.data_output
```
c. 与b部分类似，定义 export_model 接口，以便 fate-flow 可以在需要时通过它获取输出的模型。应为同时包含 “Meta” 和 “Param” 包含了产生的proto buffer类的 dict 格式。这里展示了如何导出模型。
```
    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result
```
## 开始建模任务

这里给出开发完成后，启动建模任务的一个简单示例。

#### 1：上传数据
在开始任务之前，您需要加载来自所有提供者的数据。为此，需要准备 `load_file` 配置，然后运行以下命令：
```
> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_test/upload_data.json
```
注意：每个数据提供节点（即来宾和主机）都需要执行此步骤。

#### 2：开始建模任务
在此步骤中，应准备两个与 dsl 配置文件和组件配置文件相对应的配置文件。请确保配置文件中的 `table_name` 和`namespace`与 `upload_data conf` 匹配。然后运行以下命令：
```
> python ${your_install_path}/fate_flow/fate_flow_client.py -f submitJob -d dsl_test/test_homolr_job_dsl.json -c dsl_test/${your_component_conf_json}
```
#### 3：检查日志文件
现在，您可以在以下路径中检查日志：
`${your_install_path}/logs/{your jobid}` 。

有关 dsl 配置文件和参数配置文件的更多详细信息，请参考此处的[示例文档](../ examples/federatedml-1.x-examples)中查看。