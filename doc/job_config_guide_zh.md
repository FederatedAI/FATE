### JOB RUNTIME CONFIG配置说明

Job Runtime Conf用于设置各个参与方的信息,任务的参数及各个组件的参数。 内容包括如下：

#### （1）initiator：

- **简要含义：**任务发起方的role和party_id。
- **参考配置：**

```json
"initiator": {
    "role": "guest",     
    "party_id": 9999 
}
```

#### （2）role

- **简要含义：**各参与方的信息。
- **说明：**在 role 字段中，每一个元素代表一种角色以及承担这个角色的 party_id。每个角色的 party_id 以列表形式存在，因为一个任务可能涉及到多个 party 担任同一种角色。
- **参考配置：**

```json
"role": { 
    "guest": [9999], 
    "host": [10000], 
    "arbiter": [10000]
} 
```

#### (3)job_parameters

- **简要含义：**job级别的任务配置参数

- **参数含义**:

  | 配置项       | 默认值     | 支持值  | 说明                                                       |
  | :-------------- | :----- | :----- | ------------------------------------------------------------ |
  | job_type | train | train、predict | 任务类型                      |
  | work_mode | 0 | 0、1  | 0代表单方单机版，1代表多方分布式版本                               |
  | backend   | 0    | 0、1 | 0代表EGGROLL，1代表SPARK |
  | dsl_version | 1        | 1、2                             | dsl解析器的版本号 |
  | federated_status_collect_type | PUSH | PUSH、PULL | 多方任务状态收集模式 |
  | timeout | 604800 | 正整数 | 任务超时时间,单位秒 |
  | eggroll_run | 无| processors_per_node| eggroll计算引擎相关配置参数|
  | spark_run | 无| num-executors、executor-cores |spark计算引擎相关配置参数 |
  | rabbitmq_run | 无| queue、exchange| rabbitmq 创建queue、exchange的相关配置参数|
  | task_parallelism | 2 | 正整数 | task并行度 |
  | task_nodes | 1 | 正整数 | 使用的计算节点数 |
  | task_cores_per_node | 2 | 正整数 | 每个节点使用的CPU核数 |
  | model_id | - | - | 模型id，预测任务需要填入 |
  | model_version | - | - | 模型版本, 预测任务需要填入 |
  
- **未开放参数**:

  | 配置项            | 默认值     | 支持值  | 说明                                                       |
  | :--------------- | :----- | :----- | ------------------------------------------------------------ |
  | computing_engine | 依据work_mode和backend, 自动得到 | EGGROLL、SPARK、STANDALONE | 计算引擎类型               |
  | storage_engine   | 依据work_mode和backend, 自动得到 | EGGROLL、HDFS、STANDALONE  | 组件输出中间数据存储引擎类型 |
  | federation_engine| 依据work_mode和backend, 自动得到 | EGGROLL、RABBITMQ、STANDALONE | 通信引擎类型 |
  | federated_mode   | 依据work_mode和backend, 自动得到 | SINGLE、MULTIPLE     | 实际联邦合作模式: 多方或者单方模拟多方              |
 
  **说明**:
    1. 三大类引擎具有一定的支持依赖关系，例如Spark计算引擎当前仅支持使用HDFS作为中间数据存储引擎
    2. work_mode + backend会自动依据支持依赖关系，产生对应的三大引擎配置computing、storage、federation
    3. 开发者可自行实现适配的引擎，并在runtime config配置引擎

- **参考配置：**

```json
"job_parameters": {
	"job_type": "train",
    "work_mode": 1,
    "backend": 0,
    "dsl_version": 2,
    "federated_mode": "MULTIPLE",
    "federated_status_collect_type": "PUSH",
    "timeout": 36000,
    "task_parallelism": 2,
    "task_cores": 4,
    "spark_run": {
        "num-executors": 1,
        "executor-cores": 2
    },
    "rabbitmq_run": {
        "queue": {
            "durable": true
        },
        "connection": {
            "heartbeat": 10000
        }
    }
}
```

#### (4)role_parameters

- **简要含义: **各参与方的组件参数
- **说明：**这一部分的参数对于不同的 party 都有所区别。同样地，每一个参数也是用列表的方式呈现。在  role_parameters 中，party 名被作为每一项元素的 key，而 value 则是具体提的参数内容。
- **参考配置：**

```json
"role_parameters": {
    "guest": {
        "args": {
            "data": {
                "train_data": [
                    {
                        "name": "xxxx",
                        "namespace": "xxxx"
                    }
                ]
            }
        },
        "dataio_0": {
            "with_label": [
                true
            ],
            "label_type": [
                "int"
            ],
            "output_format": [
                "dense"
            ]
        }
    },
    "host": {
        "args": {
            "data": {
                "train_data": [
                    {
                        "name": "xxxx",
                        "namespace": "xxxx"
                    }
                ]
            }
        },
        "dataio_0": {
            "with_label": [
                false,
                false
            ],
            "output_format": [
                "dense",
                "dense"
            ]
        }
    }
},
```

​                                 

#### (5) algorithm_parameters

- **简要含义：**各参与方共享的参数；
- **说明：**和上一个部分一样，在 algorithm_parameters 中，每一个参数的 key 都是在 DSL 配置文件中定义好的组件名。 在完成这些配置文件并提交任务之后，FATE-flow 将会把 role_parameters 和 algorithm_parameters 中的所有参数合并。如果合并之后，仍然存在没有定义的参数，FATE-flow 则会使用默认值。FATE-flow 会将这些参数分发到对应的 party，并开始 job。 

```json
"algorithm_parameters": {
	"intersect_0": {
    	"intersect_method": "raw",
        "sync_intersect_ids": true,
        "only_output_key": false
    },
    "hetero_lr_0": {
        "penalty": "L2",
        "optimizer": "rmsprop",
        "alpha": 0.01,
        "max_iter": 3,
        "batch_size": 320,
        "learning_rate": 0.15,
        "init_param": {
			"init_method": "random_uniform"
        }
    }
}
```
