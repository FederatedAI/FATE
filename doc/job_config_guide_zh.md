### JOB CONFIG配置说明

Job Runtime Conf用于设置各个参与方的信息,任务的参数及各个组件的参数。 内容包括如下：

#### （1）initiator：

- **简要含义：**任务发起方的role和party_id。
- **参考配置：**

```json
"initiator":{
    "role": "guest",     
    "party_id": 9999 
}
```

#### （2）role

- **简要含义：**各参与方的信息。
- **说明：**在 role 字段中，每一个元素代表一种角色以及承担这个角色的 party_id。每个角色的 party_id 以列表形式存在，因为一个任务可能涉及到多个 party 担任同一种角色。
- **参考配置：**

```json
"role":{ 
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
  | work_mode | 0 | 0、1  | 0代表单机，1代表分布式                                |
  | backend   | 0    | 0、1 | 0代表EGGROLL，1代表SPARK |
  | federated_mode               | MULTIPLE | SINGLE、MULTIPLE                 | 联邦合作模式               |
  | computing_engine | EGGROLL | EGGROLL、SPARK | 计算引擎类型               |
  | storage_engine | EGGROLL | STANDALONE、EGGROLL、HDFS、MYSQL | 存储引擎类型 |
  | engines_address | 见示例 | - | 各个引擎的地址 |
  | dsl_version | 1        | 1、2                             | dsl解析器的版本号 |
  | federated_status_collect_type | PUSH | PUSH、PULL | 状态收集模式 |
  | timeout | 604800 | 正整数 | 任务超时时间,单位秒 |
  | task_parallelism | 2 | 正整数 | task并行度 |
  | task_nodes | 1 | 正整数 | 使用的计算节点数 |
  | task_cores_per_node | 2 | 正整数 | 每个节点使用的CPU核数 |
  | model_id | - | - | 模型id，预测任务需要填入 |
  | model_version | - | - | 模型版本, 预测任务需要填入 |

  

​    **说明：**以下配置的参数仅允许配置一种:

1. work_mode + backend
2. federation_mode + computing_engine

- **参考配置：**

```json
job_parameters:{
	"job_type": "train",
    "work_mode": 1,
    "backend": 0,
    "dsl_version": 2,
    "computing_engine": "EGGROLL",
    "federation_engine": "EGGROLL",
    "storage_engine": "EGGROLL",
    "engines_address": {
        "computing": {
            "host": "127.0.0.1",
            "port": 9370
        },
        "federation": {
            "host": "127.0.0.1",
            "port": 9370
        },
        "storage": {
            "host": "172.0.0.1",
            "port": 9370
        }
    },
    "federated_mode": "MULTIPLE",
    "federated_status_collect_type": "PUSH",
    "timeout": 36000,
    "task_parallelism": 2,
    "task_nodes": 1,
    "task_cores_per_node": 2,
    "task_memory_per_node": 512,
    "model_id": "arbiter-10000#guest-9999#host-9999_10000#model",
    "model_version": "2020092416160711633252"
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

