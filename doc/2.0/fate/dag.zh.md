# DAG使用指南
## 说明
本文档主要说明FATE-v2.0以后，如何使用DAG去定义一个联邦学习建模任务

## DAG定义
该章节主要介绍DAG文件的字段定义，在fate-v2.0，使用yaml文件格式去描述DAG

DAG例子
```yaml
dag:
  parties:
  - party_id: ['9999']
    role: guest
  - party_id: ['10000']
    role: host
  party_tasks:
    guest_9999:
      parties:
      - party_id: ['9999']
        role: guest
      tasks:
        reader_0:
          parameters: {name: breast_hetero_guest, namespace: experiment}
    host_10000:
      parties:
      - party_id: ['10000']
        role: host
      tasks:
        reader_0:
          parameters: {name: breast_hetero_host, namespace: experiment}
  stage: train
  tasks:
    eval_0:
      component_ref: evaluation
      dependent_tasks: [sbt_0]
      inputs:
        data:
          input_datas:
            task_output_artifact:
            - output_artifact_key: train_output_data
              parties:
              - party_id: ['9999']
                role: guest
              producer_task: sbt_0
      parameters:
        label_column_name: null
        metrics: [auc]
        predict_column_name: null
      parties:
      - party_id: ['9999']
        role: guest
      stage: default
    psi_0:
      component_ref: psi
      dependent_tasks: [reader_0]
      inputs:
        data:
          input_data:
            task_output_artifact:
              output_artifact_key: output_data
              parties:
              - party_id: ['9999']
                role: guest
              - party_id: ['10000']
                role: host
              producer_task: reader_0
      parameters: {}
      stage: default
    reader_0:
      component_ref: reader
      parameters: {}
      stage: default
    sbt_0:
      component_ref: hetero_secureboost
      dependent_tasks: [psi_0]
      inputs:
        data:
          train_data:
            task_output_artifact:
              output_artifact_key: output_data
              parties:
              - party_id: ['9999']
                role: guest
              - party_id: ['10000']
                role: host
              producer_task: psi_0
        model: {}
      parameters:
        max_depth: 3
        num_trees: 2
        ...
schema_version: 2.1.0
kind: fate
```

### 一级字段
#### dag
表示的是整个dag的描述文件

#### schema_version
保留字段，为对应的版本号

#### kind
协议类型，fate-v2.0的dag为通用联邦学习任务工作流描述文件，支持多种协议，默认为fate-v2.0协议，即"fate"，如果需要使用fate-v2.0的dag来表示其他协议，修改该字段即可

### dag下二级字段
二级字段由五个组件：分别是parties，conf, stage, tasks, party_tasks.

class DAGSpec(BaseModel):
    tasks: Dict[str, TaskSpec]
    party_tasks: Optional[Dict[str, PartyTaskSpec]]
### parties
parties代表参与方，是一个列表，其中列表的每个元素为一个party描述对象，单个party对象由role和party_id组成

```yaml
- party_id: ['9999']
  role: guest
```

##### role
role表示联邦建模中的角色，常用的是"guest", "host", "arbiter"

##### party_id
是一个列表，代表该角色下有哪些party_id组成

#### conf
job级别的任务参数，具体可以参考[文档](https://github.com/FederatedAI/FATE-Flow/blob/master/doc/job_scheduling.zh.md)

#### stage
表示job运行的阶段，可能值为"default","train","predict","cross_validation"之一。  
* default: 适用于一些无需区分运行阶段的组件，如psi,union,evaluation等
* train: 适用于特征工程、训练算法等，表示该组件是运行的训练过程
* predict: 使用于特征工程、训练算法等，表示该组件是运行的预测流程
* cross_validation: 用于训练算法，表示需要做交叉验证

注意的是，stage为job级别的状态，当具体任务运行阶段不指定时，会继承job级别的状态，否则，以任务内部指定的为准

#### tasks
通用任务配置，是一个字典格式，key为任务名称，value为该任务的通用描述，下级字段为component_ref、dependent_task、parameters、inputs、outputs、parties、conf、stage组成。

##### component_ref
该任务需要调用的组件引用名称，为一个字符串，组件说明可以参考[组件描述文档](./components/README.zh.md)

##### dependent_tasks
该任务的上游依赖任务，为一个列表

```yaml
dependent_tasks: [psi_0]
```

##### parameters
该任务的通用参数配置

```yaml
parameters:
  max_depth: 3
  num_trees: 2
  ...
```

##### inputs
该任务的输入，包括上游任务输出作为输入、模型仓库直接输入、数据仓库直接输入(备用字段)。
inputs: 类型为字典，字典的key为data或者model，value为端口连线关系。
端口连线关系：类型为字典，字典的key为该任务对应的组件的输入key，如train_data、validate_data、input_data等，value是一个输入描述对象。
输入描述对象 上游任务输出作为输入、模型仓库直接输入、数据仓库直接输入

###### 上游任务输出作为输入
使用task_output_artifact代表上游任务输出作为输入
* producer_tasks: 上游任务的名称
* output_artifact_key: 上游任务的具体哪个输出口作为输入
* parties: 选填，表示该参与方使用该输入，默认不指定，如果不指定，任务的每一方输入一致，否则，以指定的为准，用户可以通过parties来控制多方不对称输入

```yaml
eval_0:
      component_ref: evaluation
      dependent_tasks: [sbt_0]
      inputs:
        data:
          input_datas:
            task_output_artifact:
            - output_artifact_key: train_output_data
              parties:
              - party_id: ['9999']
                role: guest
              producer_task: sbt_0
```

###### 模型仓库直接输入
使用model_warehouse代表从模型仓库直接输入，当前主要用于预测阶段的模型输入

model_id: 模型id
model_version: 模型版本
producer_task: (model_id, model_version) 对应的训练任务的名称
output_artifact_key: (model_id, model_version) 对应的训练任务的模型输出名
parties: 选填，表示该参与方使用该输入，默认不指定，如果不指定，任务的每一方输入一致，否则，以指定的为准，用户可以通过parties来控制多方不对称输入 

##### outputs
选填，表示组件的输出，fate-v2.0协议调度时不需要填写，当前仅用于互联互通协议的配置

当前输出的类型为字典，字典的key包括data、model、metric输出，value格式包含三个字段output_artifact_key_alias，output_artifact_key_alias,parties。

##### parties
该任务参与方，格式和job级别的parties配置一致，如果用户填写，则该任务运行时的参与方为用户填写值，否则，继承job级别的

##### conf
任务级别的配置，为一个字典

##### stage
任务运行的阶段，如果填写则以填写为准，不填则继承job级别的

#### party_tasks
party_tasks表示各方的个性化配置，为一个字典，字典的key为别称，表示哪些站点会使用该个性化设置，value为parties、tasks、conf。

```yaml
guest_9999:
  parties:
  - party_id: ['9999']
    role: guest
  tasks:
    reader_0:
      parameters: {name: breast_hetero_guest, namespace: experiment}
```
##### parties
表示有哪些参与方

##### conf
字典类型，表示对应的parties运行时tasks对应任务时的个性化任务配置

#### tasks
字典类型，key为有哪些任务，value为具体任务对应的配置，配置由conf和parameters构成，
其中conf是一个字典，表示的是parties对应的参与方运行这个具体任务的配置，parameters是这些参与方运行这个具体任务时的算法参数