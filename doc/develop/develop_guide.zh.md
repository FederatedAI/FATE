# 开发指南

## 为 FATE 开发可运行的算法模块

本文档描述了如何开发算法模块，使得该模块可以在 FATE 架构下被调用。

要开发模块，需要执行以下 5 个步骤。

1.  定义将在此模块中使用的 python 参数对象。
2.  定义模块的 meta 文件。
3.  如果模块需要联邦，则需定义传输变量配置文件。
4.  您的算法模块需要继承model\_base类，并完成几个指定的函数。
5.  定义模型保存所需的protobuf文件。
6.  若希望通过python脚本直接启动组件，需要在fate\_client中定义Pipeline组件。

在以下各节中，我们将通过 toy\_example 详细描述这 5 个步骤。

### 第 1 步：定义此模块将使用的参数对象

参数对象是将用户定义的运行时参数传递给开发模块的唯一方法，因此每个模块都有其自己的参数对象。

为定义可用的参数对象，需要三个步骤。

1.  打开一个新的 python 文件，将其重命名为 xxx\_param.py，其中xxx代表您模块的名称，并将其放置在
    <span class="title-ref">python/federatedml/param/</span> 文件夹中。 在
    xxx\_param.py 中定义它的类对象，应该继承
    <span class="title-ref">python/federatedml/param/base\_param.py</span>
    中定义的 BaseParam 类。
2.  参数类的 <span class="title-ref">\_\_init\_\_</span> 方法应该指定模块使用的所有参数。
3.  重载 BaseParam 的参数检查接口，否则将会抛出未实现的错误。检查方法被用于验证参数变量是否可用。

以 hetero lr 的参数对象为例，python文件为
`federatedml/param/logistic\_regression\_param.py`
<!-- {% include-example "../../python/federatedml/param/logistic_regression_param.py" %} -->

首先，它继承自 BaseParam：

```python
class LogisticParam(BaseParam):
```

然后，在 <span class="title-ref">\_\_init\_\_</span> 方法中定义所有参数变量：

```python
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

如上面的示例所示，该参数也可以是 Param
类。此类参数的默认设置是此类的一个实例。然后将该实例的深度复制（\`deepcopy\`）版本分配给类归属。深度复制功能用于避免任务同时运行时指向相同内存的风险。

一旦正确定义了类，已有的参数解析器就可以递归地解析每个属性的值。

之后，重载参数检查的接口：

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

### 第二步：定义新模块的meta文件

定义meta文件是为了使 FATE-Flow
模块通过该文件以获取有关如何启动模块程序的信息。

1.  在
    [components](../../python/federatedml/components)
    中定义名为 xxx.py 的meta文件，其中 xxx 是您要开发的模块。
    
2.  配置 meta 文件。
       - 继承 ComponentMeta, 用模块名为其命名, 
      例如 xxx_cpn_meta = ComponentMeta("XXX"). XXX 即在 dsl 中调用的模块名。
      ``` sourceCode python
          from .components import ComponentMeta
          hetero_lr_cpn_meta = ComponentMeta("HeteroLR")
      ``` 
      - 使用装饰器 `xxx_cpn_meta.bind_runner.on_$role`将模块object绑定至每个角色。
        $role 包括 guest\host\arbiter. 如果多个角色使用同一模块object，可以使用
          `xxx_cpn_meta.bind_runner.on_$role1.on_$role2.on_$role3` 格式注明。 
        装饰器方程将引入并返回对应角色的模块object。
   
        以hetero-lr 为例：
        [python/federatedml/components/hetero_lr.py](../../python/federatedml/conf/setting_conf/HeteroLR.json)
        
        ``` sourceCode python
            @hetero_lr_cpn_meta.bind_runner.on_guest
            def hetero_lr_runner_guest():
                from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_guest import HeteroLRGuest
                
                return HeteroLRGuest
                
            @hetero_lr_cpn_meta.bind_runner.on_host
            def hetero_lr_runner_host():
                from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_host import HeteroLRHost
                
                return HeteroLRHost
        ``` 
        - 使用装饰器 `xxx_cpn_meta.bind_param` 将参数object绑定至step1中定义的各个模块object，
        装饰器方程将引入并返回对应参数object。
          
          ``` sourceCode python
              @hetero_lr_cpn_meta.bind_param
              def hetero_lr_param():
                  from federatedml.param.logistic_regression_param import HeteroLogisticParam
                  
                  return HeteroLogisticParam
          ``` 

### 第三步：定义此模块的传递变量py文件并生成传递变量对象（可选）

仅在此模块被联邦时（即不同参与方之间存在信息交互）才需要执行此步骤。

<div class="note">

<div class="admonition-title">

Note

</div>

应将其放在
[transfer\_class](../../python/federatedml/transfer_variable/transfer_class)
文件夹中。

</div>

在该定义文件中，您需要创建需要的 transfer\_variable
类，并继承BaseTransferVariables类，然后定义相应的变量，并为其赋予需要的传输权限。以
“HeteroLRTransferVariable”为例，可以参考以下代码：

``` sourceCode python
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

其中，需要设定的属性为：

  - name  
    变量名

  - src  
    应为 "guest"，"host"，"arbiter" 的某些组合，它表示发送交互信息从何处发出。

  - dst  
    应为 "guest"，"host"，"arbiter" 的某些组合列表，用于定义将交互信息发送到何处。

在 python 文件编写完成后，运行以下程序，可在
[auth\_conf](../../python/federatedml/transfer_variable/auth_conf)
中生成对应的json配置文件。该配置文件将被fate\_flow识别并用于后续权限判断。

``` sourceCode bash
python fate_arch/federation/transfer_variable/scripts/generate_auth_conf.py federatedml federatedml/transfer_variable/auth_conf
```

### 第四步：定义您的模块（应继承 model\_base）

fate\_flow\_client 模块的运行规则是：

1.  检索 setting\_conf 并找到配置文件的“module”和“role”字段。
2.  初始化各方的运行对象。
3.  调用运行对象的 run 方法。
4.  如果需要，调用 save\_data 方法。
5.  如果需要，调用 export\_model 方法。

在本节中，我们讲解如何执行规则 3 至 5
。需要被继承的model\_base类位于：[python/federatedml/model\_base.py](../../python/federatedml/model_base.py)
.

  - 在需要时重载 fit 接口  
    fit 函数具有以下形式。
    
    ``` sourceCode python
    def fit(self, train_data, validate_data=None):
    ```
    
    > fit
    > 函数是启动建模组件的训练，或者特征工程组件的fit功能的入口。接受训练数据和验证集数据，validate数据可不提供。该函数在用户启动训练任务时，被model\_base自动调起，您只需在该函数完成自身需要的fit任务即可。

  - 在需要的时候重载 predict 接口  
    predict 函数具有如下形式.
    
    ``` sourceCode python
    def predict(self, data_inst):
    ```
    
    Data\_inst 是一个 Table.
    用于建模组件的预测功能。在用户启动预测任务时，将被model\_base自动调起。另外，在训练任务中，建模组件也会调用predict函数对训练数据和验证集数据（如果有）进行预测，并输出预测结果。该函数的返回结果，如果后续希望接入evaluation，需要输出符合下列格式的Table：
    
    >   -   - 二分类，多分类，回归任务: \["label", "predict\_result",
    >         "predict\_score", "predict\_detail", "type"\]
    >         
    >           - label:提供的标签
    >           - predict\_result: 模型预测的结果
    >           - predict\_score:
    >             对于2分类为1的预测分数，对于多分类为概率最高的那一类的分数，对于回归任务，与predict\_result相同
    >           - predict\_detail: 对于分类任务，列出各分类的得分，对于回归任务，列出回归预测值
    >           - type: 表明该结果来源（是训练数据或者是验证及数据）,该结果model\_base会自动拼接。
    > 
    >   -   - 聚类任务返回两张表  
    >         第一张的格式为: \["cluster\_sample\_count",
    >         "cluster\_inner\_dist", "inter\_cluster\_dist"\]
    >           - cluster\_sample\_count: 每个类别下的样本个数
    >           - cluster\_inner\_dist: 类内距离
    >         \* inter\_cluster\_dist: 类间距离 第二张表的格式为:
    >         \["predicted\_cluster\_index", "distance"\]
    >           - predicted\_cluster\_index: 预测的所属类别
    >           - distance: 该样本到中心点的距离

  - 在需要的时候重载 transform 接口  
    transform 函数具有如下形式.
    
    ``` sourceCode python
    def transform(self, data_inst):
    ```
    
    Data\_inst 是一个 Table.
    用于特征工程组件对数据进行转化功能。在用户启动预测任务时，将被model\_base自动调起。

  - 定义您的 save\_data 接口  
    以便 fate-flow 可以在需要时通过它获取输出数据。
    
    ``` sourceCode python
    def save_data(self):
        return self.data_output
    ```

### 第五步： 定义模型保存所需的protobuf

  - 定义模型保存所需的protobuf文件:  
    为了方便模型跨平台保存和读取模型，FATE使用protobuf文件定义每个模型所需的参数和模型内容。当您开发自己的模块时，需要定义本模块中需要保存的内容并创建相应的protobuf文件。protobuf文件所在的位置为
    [这个目录](../../python/federatedml/protobuf/proto) 。

更多使用protobuf的细节，请参考
[这个教程](https://developers.google.com/protocol-buffers/docs/pythontutorial)

每个模型一般需要两个proto文件，其中后缀为meta的文件中保存某一次任务的配置，后缀为param的文件中保存某次任务的模型结果。

在完成proto文件的定义后，可执行protobuf目录下的
[generate\_py.sh文件](../../python/fate_arch/protobuf/generate_py.sh)
生成对应的python文件。之后，您可在自己的项目中引用自己设计的proto类型，并进行保存：

> 
> 
> ``` sourceCode bash
> bash proto_generate.sh
> ```

  - 定义 export\_model 接口  
    以便 fate-flow 可以在需要时通过它获取输出的模型。应为同时包含 “Meta” 和 “Param” 包含了产生的proto
    buffer类的 dict 格式。这里展示了如何导出模型。
    
    ``` sourceCode python
    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result
    ```

### 第六步：开发Pipeline组件

若希望后续用户可以通过python脚本形式启动建模任务，需要在
[python/fate\_client/pipeline/component](../../python/fate_client/pipeline/component)
中添加自己的组件。详情请参考Pipeline的
[文档](../api/fate_client/pipeline.md)

## 开始建模任务

这里给出开发完成后，启动建模任务的一个简单示例。

  - 1\. 上传数据  
    在开始任务之前，您需要加载来自所有提供者的数据。为此，需要准备
    <span class="title-ref">load\_file</span> 配置，然后运行以下命令：

<!-- end list -->

``` sourceCode bash
flow data upload -c upload_data.json
```

..Note:: 每个数据提供节点（即来宾和主机）都需要执行此步骤。

  - 2\. 开始建模任务  
    在此步骤中，应准备两个与 dsl 配置文件和组件配置文件相对应的配置文件。请确保配置文件中的
    <span class="title-ref">table\_name</span> 和\`namespace\`与
    <span class="title-ref">upload\_data conf</span> 匹配。然后运行以下命令：

<!-- end list -->

``` sourceCode bash
flow job submit -d ${your_dsl_file.json} -c ${your_component_conf_json}
```

> 若您已在fate\_client中添加了自己的组件，也可以准备好自己的pipeline脚本，然后使用python命令直接启动：

``` sourceCode bash
python ${your_pipeline.py}
```

  - 3.检查日志文件  
    现在，您可以在以下路径中检查日志：\`${your\_install\_path}/logs/{your jobid}\`.

有关 dsl 配置文件和参数配置文件的更多详细信息，请参考此处的\`examples/dsl/v2\`中查看。
