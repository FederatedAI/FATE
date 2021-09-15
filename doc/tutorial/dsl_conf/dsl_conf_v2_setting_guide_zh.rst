任务配置和运行配置 V2
=====================

[`ENG`_]

.. _ENG: dsl_conf_v2_setting_guide.rst

关于DSL V1的说明，请参考[`V1`_]

.. _V1: dsl_conf_v1_setting_guide_zh.rst

DSL 配置说明
------------

1. 概述
~~~~~~~

DSL 的配置文件采用 json 格式，实际上，整个配置文件就是一个 json 对象 （dict）。

2. Components
~~~~~~~~~~~~~~

-  **含义：** 在这个 dict 的第一级是 "components"，用来表示这个任务将会使用到的各个模块。
-  **参考：**

.. code-block:: json

  {
    "components" : {
            ...
        }
    }

-  **说明：**

每个独立的模块定义在 "components" 之下，例如：

.. code-block:: json

  "dataio_0": {
        "module": "DataIO",
        "input": {
            "data": {
                "data": [
                    "reader_0.train_data"
                ]
            }
        },
        "output": {
            "data": ["train"],
            "model": ["dataio"]
        }
    }

所有数据需要通过**Reader**模块从数据存储拿取数据，注意此模块仅有输出``output``

.. code-block:: json

  "reader_0": {
        "module": "Reader",
        "output": {
            "data": ["train"]
        }
  }

3. 模块
~~~~~~~

-  **含义：** 用来指定使用的模块。这
-  **说明：** 个参数的内容需要和 `federatedml/conf/setting_conf` 下各个模块的文件名保持一致（不包括 .json 后缀）。
-  **参考：**

.. code:: json

   "hetero_feature_binning_1": {
       "module": "HeteroFeatureBinning",
        ...
   }

4. 输入
~~~~~~~~

-  **含义：** 上游输入，分为两种输入类型，分别是数据和模型。

4.1 数据输入
^^^^^^^^^^^^^^^

-  **含义：** 上游数据输入，分为三种输入类型：

    1. data: 一般被用于 data_io 模块, feature_engineering 模块或者 evaluation 模块
    2. train_data: 一般被用于 homo_lr, hetero_lr 和 secure_boost 模块。如果出现了 train_data 字段，那么这个任务将会被识别为一个 fit 任务
    3. validate_data： 如果存在 train_data 字段，那么该字段是可选的。如果选择保留该字段，则指向的数据将会作为 validation set
    4. test_data: 用作预测数据，如提供，需同时提供model输入。

4.2 模型输入
^^^^^^^^^^^^^^^^
-  **含义：** 上游模型输入，分为两种输入类型：

   1. model: 用于同种类型组件的模型输入。例如，hetero_binning_0 会对模型进行 fit，然后 hetero_binning_1 将会使用 hetero_binning_0 的输出用于 predict 或 transform。代码示例：

      .. code-block:: json

          "hetero_feature_binning_1": {
              "module": "HeteroFeatureBinning",
              "input": {
                  "data": {
                      "data": [
                          "dataio_1.validate_data"
                      ]
                  },
                  "model": [
                      "hetero_feature_binning_0.fit_model"
                  ]
              },
              "output": {
                  "data": ["validate_data"],
                "model": ["eval_model"]
              }
          }

   2. isometric_model: 用于指定继承上游组件的模型输入。 例如，feature selection 的上游组件是 feature binning，它将会用到 feature binning 的信息来作为 feature importance。代码示例：

      .. code-block:: json

            "hetero_feature_selection_0": {
                "module": "HeteroFeatureSelection",
                "input": {
                    "data": {
                        "data": [
                            "hetero_feature_binning_0.train"
                        ]
                    },
                    "isometric_model": [
                        "hetero_feature_binning_0.output_model"
                    ]
                },
                "output": {
                    "data": ["train"],
                    "model": ["output_model"]
                }
            }

5. 输出
~~~~~~~~

-  **含义：** 输出，与输入一样，分为数据和模型输出

5.1 数据输出
^^^^^^^^^^^^^^^

-  **含义：** 数据输出，分为四种输出类型：

1. data: 常规模块数据输出
2. train_data: 仅用于Data Split
3. validate_data: 仅用于Data Split
4. test_data： 仅用于Data Split

5.2 模型输出
^^^^^^^^^^^^^^^^
-  **含义：** 模型输出，仅使用model


JOB RUNTIME CONFIG配置说明，针对1.5.x版本新格式
-----------------------------------------------

1. 概述
~~~~~~~

Job Runtime Conf用于设置各个参与方的信息, 作业的参数及各个组件的参数。
内容包括如下：

2. DSL版本
~~~~~~~~~~

-  **含义：** 配置版本，默认不配置为1，建议配置为2
-  **参考：**

.. code:: json

   "dsl_version": "2"

3. 作业参与方
~~~~~~~~~~~~~

3.1 发起方
^^^^^^^^^^

-  **含义：** 任务发起方的role和party_id。
-  **参考：**

.. code:: json

   "initiator": {
       "role": "guest",
       "party_id": 9999
   }

3.2 所有参与方
^^^^^^^^^^^^^^

-  **含义：** 各参与方的信息。
-  **说明：** 在 role 字段中，每一个元素代表一种角色以及承担这个角色的
   party_id。每个角色的 party_id
   以列表形式存在，因为一个任务可能涉及到多个 party 担任同一种角色。
-  **参考：**

.. code:: json

   "role": {
       "guest": [9999],
       "host": [10000],
       "arbiter": [10000]
   }

4. 系统运行参数
~~~~~~~~~~~~~~~

-  **含义：** 配置作业运行时的主要系统参数

4.1 参数应用范围策略设置
^^^^^^^^^^^^^^^^^^^^^^^^

-  应用于所有参与方，使用common范围标识符
-  仅应用于某参与方，使用role范围标识符，使用\ :math:`role:`\ party_index定位被指定的参与方，直接指定的参数优先级高于common参数

.. code:: json

   "common": {
   }

   "role": {
     "guest": {
       "0": {
       }
     }
   }

其中common下的参数应用于所有参与方，role-guest-0配置下的参数应用于guest角色0号下标的参与方
注意，当前版本系统运行参数未对仅应用于某参与方做严格测试，建议使用优先选用common

4.2 支持的系统参数
^^^^^^^^^^^^^^^^^^

.. list-table:: 支持的系统参数
   :widths: 20 20 30 30
   :header-rows: 1

   * - 配置项
     - 默认值
     - 支持值
     - 说明

   * - job_type
     - train
     - train, predict
     - 任务类型

   * - work_mode
     - 0
     - 0, 1
     - 0代表单方单机版，1代表多方分布式版本

   * - backend
     - 0
     - 0, 1, 2
     - 0代表EGGROLL，1代表SPARK加RabbitMQ，2代表SPARK加Pulsar

   * - task_cores
     - 4
     - 正整数
     - 作业申请的总cpu核数

   * - task_parallelism
     - 1
     - 正整数
     - task并行度

   * - computing_partitions
     - task所分配到的cpu核数
     - 正整数
     - 计算时数据表的分区数

   * - eggroll_run
     - 无
     - processors_per_node等
     - eggroll计算引擎相关配置参数，一般无须配置，由task_cores自动计算得到，若配置则task_cores参数不生效

   * - spark_run
     - 无
     - num-executors, executor-cores等
     - spark计算引擎相关配置参数，一般无须配置，由task_cores自动计算得到，若配置则task_cores参数不生效

   * - rabbitmq_run
     - 无
     - queue, exchange等
     - rabbitmq创建queue、exchange的相关配置参数，一般无须配置，采取系统默认值

   * - pulsar_run
     - 无
     - producer, consumer等
     - pulsar创建producer和consumer时候的相关配置，一般无需配置。

   * - federated_status_collect_type
     - PUSH
     - PUSH, PULL
     - 多方运行状态收集模式，PUSH表示每个参与方主动上报到发起方，PULL表示发起方定期向各个参与方拉取

   * - timeout
     - 259200 (3天)
     - 正整数
     - 任务超时时间,单位秒

   * - model_id
     - \-
     - \-
     - 模型id，预测任务需要填入

   * - model_version
     - \-
     - \-
     - 模型version，预测任务需要填入

.. note::

1. 三大类引擎具有一定的支持依赖关系，例如Spark计算引擎当前仅支持使用HDFS作为中间数据存储引擎
2. work_mode + backend会自动依据支持依赖关系，产生对应的三大引擎配置computing、storage、federation
3. 开发者可自行实现适配的引擎，并在runtime config配置引擎

4.3 未开放参数
^^^^^^^^^^^^^^


.. list-table:: 未开放参数
   :widths: 20 20 30 30
   :header-rows: 1

   * - 配置项
     - 默认值
     - 支持值
     - 说明

   * - computing_engine
     - 依据work_mode和backend, 自动得到
     - EGGROLL, SPARK, STANDALONE
     - 计算引擎类型

   * - storage_engine
     - 依据work_mode和backend, 自动得到
     - EGGROLL, HDFS, STANDALONE
     - 组件输出中间数据存储引擎类型

   * - federation_engine
     - 依据work_mode和backend, 自动得到
     - EGGROLL, RABBITMQ, STANDALONE, PULSAR
     - 通信引擎类型

   * - federated_mode
     - 依据work_mode和backend, 自动得到
     - SINGLE, MULTIPLE
     - 联邦合作模式: 多站点多方或者单站点模拟多方


4.4 参考配置
^^^^^^^^^^^^

1. 使用eggroll作为backend，采取默认cpu分配计算策略时的配置

.. code:: json

   "job_parameters": {
     "common": {
       "job_type": "train",
       "work_mode": 1,
       "backend": 0,
       "task_cores": 6,
       "task_parallelism": 2,
       "computing_partitions": 8,
       "timeout": 36000
     }
   }

2. 使用eggroll作为backend，采取直接指定cpu等参数时的配置

.. code:: json

   "job_parameters": {
     "common": {
       "job_type": "train",
       "work_mode": 1,
       "backend": 0,
       "eggroll_run": {
         "eggroll.session.processors.per.node": 2
       },
       "task_parallelism": 2,
       "computing_partitions": 8,
       "timeout": 36000,
     }
   }

3. 使用spark加rabbitMQ作为backend，采取直接指定cpu等参数时的配置

.. code:: json

   "job_parameters": {
     "common": {
       "job_type": "train",
       "work_mode": 1,
       "backend": 1,
       "spark_run": {
         "num-executors": 1,
         "executor-cores": 2
       },
       "task_parallelism": 2,
       "computing_partitions": 8,
       "timeout": 36000,
       "rabbitmq_run": {
         "queue": {
           "durable": true
         },
         "connection": {
           "heartbeat": 10000
         }
       }
     }
   }
4. 使用spark加pulsar作为backend

.. code::json

   "job_parameters": {
     "common": {
       "work_mode": 1,
       "backend": 2,
       "spark_run": {
         "num-executors": 1,
         "executor-cores": 2
       },
     }
   }

4.5 资源管理详细说明
^^^^^^^^^^^^^^^^^^^^

1.5.0版本开始，为了进一步管理资源，fateflow启用更细粒度的cpu
cores管理策略，去除早前版本直接通过限制同时运行作业个数的策略

4.5.1 总资源配置
''''''''''''''''

-  资源来自于基础引擎，当前版本未实现自动获取基础引擎的资源大小，因此你通过配置文件的方式配置\ ``$PROJECT_BASE/conf/service_conf.yaml``\ 指定，fateflow
   server启动时从配置文件扫描所有基础引擎信息并注册到数据库表\ ``t_engine_registry``
-  fate_on_eggroll：total_cores=cores_per_node*nodes
-  fate_on_spark：total_cores=cores_per_node*nodes
-  standalone：使用\ ``$PROJECT_BASE/python/fate_flow/settings.py``\ 的\ **STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE**\ 虚拟配置
-  不同基础引擎间的资源计算互相隔离
-  以上配置修改后均需要重启fateflow server使之生效

4.5.2 运行资源计算
''''''''''''''''''

计算每个task实际提交到计算引擎的task_run_cores，但并不代表资源调度时的申请量

1. job conf使用task_cores配置：

   -  task_run_cores(guest, host)：max(task_cores / total_nodes, 1) \* total_nodes
   -  task_run_cores(arbiter)：max(1 / total_nodes, 1) \* total_nodes
   -  fateflow会将参数自动转换为对应引擎的实际配置参数，如eggroll的eggroll.session.processors.per.node，spark的executor-cores和num-executors

2. job conf使用eggroll_run配置：

   -  task_run_cores(guest, host, arbiter)：eggroll.session.processors.per.node \* total_nodes

3. job conf使用spark_run配置：

   -  task_run_cores(guest, host, arbiter)：executor-cores \* num-executors

4.5.3 资源调度
''''''''''''''

1. 作业申请资源的计算

   -  对于计算引擎为eggroll、standalone

      -  apply_cores(guest, host)：task_run_cores \* task_parallelism
      -  apply_cores(arbiter)：0，因为实际上仅消耗极少量资源且eggroll暂仅支持配置所有node节点cores一致，因此为了避免nodes太多导致arbiter资源扣减资源太多影响作业排队，所以资源调度计算时设为\ **0**
      -  此处注意，在eggroll集群上，arbiter依然会在每个node被分配了task_run_cores/nodes个cores

   -  对于计算引擎为spark

      -  对于spark，支持executor-cores \*
         num-executors，不与集群nodes数强相关，尤其spark本身有资源调度器，如果此处资源调度计算与实际提交不一致，可能会导致spark作业一直等待
      -  apply_cores(guest, host, arbiter)：task_run_cores \*
         task_parallelism

2. 作业调度策略

   -  按提交时间先后入队
   -  目前仅支持FIFO策略，也即每次调度器仅会扫描第一个作业，若第一个作业申请资源成功则start且出队，若申请资源失败则等待下一轮调度

3. 资源申请规则

   -  调度器依据上述调度策略选出作业，分发联邦多方资源申请指令到所有参与方
   -  若所有参与方均申请资源成功(total_cores - apply_cores >
      0)，则该作业申请资源成功
   -  若非所有参与方均申请资源成功，则发送资源回滚指令到已申请成功的参与方，该作业申请资源失败

5. 组件运行参数
~~~~~~~~~~~~~~~

5.1 参数应用范围策略设置
^^^^^^^^^^^^^^^^^^^^^^^^

-  应用于所有参与方，使用common范围标识符
-  仅应用于某参与方，使用role范围标识符，使用\ :math:`role:`\ party_index定位被指定的参与方，直接指定的参数优先级高于common参数

.. code:: json

   "commom": {
   }

   "role": {
     "guest": {
       "0": {
       }
     }
   }

其中common配置下的参数应用于所有参与方，role-guest-0配置下的参数表示应用于guest角色0号下标的参与方
注意，当前版本组件运行参数已支持两种应用范围策略


5.2 参考配置
^^^^^^^^^^^^

-  ``intersection_0``\ 与\ ``hetero_lr_0``\ 两个组件的运行参数，放在common范围下，应用于所有参与方
-  对于\ ``reader_0``\ 与\ ``dataio_0``\ 两个组件的运行参数，依据不同的参与方进行特定配置，这是因为通常不同参与方的输入参数并不一致，所有通常这两个组件一般按参与方设置
-  上述组件名称是在DSL配置文件中定义

.. code:: json

   "component_parameters": {
     "common": {
       "intersection_0": {
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
     },
     "role": {
       "guest": {
         "0": {
           "reader_0": {
             "table": {"name": "breast_hetero_guest", "namespace": "experiment"}
           },
           "dataio_0":{
             "with_label": true,
             "label_name": "y",
             "label_type": "int",
             "output_format": "dense"
           }
         }
       },
       "host": {
         "0": {
           "reader_0": {
             "table": {"name": "breast_hetero_host", "namespace": "experiment"}
           },
           "dataio_0":{
             "with_label": false,
             "output_format": "dense"
           }
         }
       }
     }
   }

5.3 多Host 配置
^^^^^^^^^^^^^^^^

多Host任务应在role下列举所有host信息

- **样例**:

.. code-block:: json

   "role": {
      "guest": [
        10000
      ],
      "host": [
        10000, 10001, 10002
      ],
      "arbiter": [
        10000
      ]
   }

各host不同的配置应在各自对应模块下分别列举

- **样例**:

.. code-block:: json

   "component_parameters": {
      "role": {
         "host": {
            "0": {
               "reader_0": {
                  "table":
                   {
                     "name": "hetero_breast_host_0",
                     "namespace": "hetero_breast_host"
                   }
               }
            },
            "1": {
               "reader_0": {
                  "table":
                  {
                     "name": "hetero_breast_host_1",
                     "namespace": "hetero_breast_host"
                  }
               }
            },
            "2": {
               "reader_0": {
                  "table":
                  {
                     "name": "hetero_breast_host_2",
                     "namespace": "hetero_breast_host"
                  }
               }
            }
         }
      }
   }

5.4 预测任务配置
^^^^^^^^^^^^^^^^

5.4.1 说明
''''''''''

DSL V2不会自动为训练任务生成预测dsl。
用户需要首先使用`Flow Client <../python/fate_client/flow_client/README.rst>`__ 部署所需模型中模块。
详细命令说明请参考`FATE-Flow document <../python/fate_client/flow_client/README.rst#deploy>`__

.. code-block:: bash

    flow model deploy --model-id $model_id --model-version $model_version --cpn-list ...

可选地，用户可以在预测dsl中加入新模块，如``Evaluation``

5.4.2 样例
''''''''''''''

训练 dsl：

.. code-block:: json

    "components": {
        "reader_0": {
            "module": "Reader",
            "output": {
                "data": [
                    "data"
                ]
            }
        },
        "dataio_0": {
            "module": "DataIO",
            "input": {
                "data": {
                    "data": [
                        "reader_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        },
        "intersection_0": {
            "module": "Intersection",
            "input": {
                "data": {
                    "data": [
                        "dataio_0.data"
                    ]
                }
            },
            "output": {
                "data":[
                    "data"
                ]
            }
        },
        "hetero_nn_0": {
            "module": "HeteroNN",
            "input": {
                "data": {
                    "train_data": [
                        "intersection_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        }
    }

预测 dsl:

.. code-block:: json

    "components": {
        "reader_0": {
            "module": "Reader",
            "output": {
                "data": [
                    "data"
                ]
            }
        },
        "dataio_0": {
            "module": "DataIO",
            "input": {
                "data": {
                    "data": [
                        "reader_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        },
        "intersection_0": {
            "module": "Intersection",
            "input": {
                "data": {
                    "data": [
                        "dataio_0.data"
                    ]
                }
            },
            "output": {
                "data":[
                    "data"
                ]
            }
        },
        "hetero_nn_0": {
            "module": "HeteroNN",
            "input": {
                "data": {
                    "train_data": [
                        "intersection_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        },
        "evaluation_0": {
            "module": "Evaluation",
            "input": {
                "data": {
                    "data": [
                        "hetero_nn_0.data"
                    ]
                }
             },
             "output": {
                 "data": [
                     "data"
                 ]
              }
        }
    }

6. 基本原理
~~~~~~~~~~~

1. 提交作业后，fateflow获取job dsl与job
   config，存于数据库\ ``t_job``\ 表对应字段以及\ ``$PROJECT_BASE/jobs/$jobid/``\ 目录
2. 解析job dsl与job
   config，依据合并参数生成细粒度参数(如上述所说的backend&work_mode对应会生成三个引擎参数),
   以及处理参数默认值
3. 将共同配置分发到各参与方并存储，依据参与方的实际信息，生成\ **job_runtime_on_party_conf**\ ，同样存于数据库与jobs目录
4. 每个参与方接收到任务时，均依据\ **job_runtime_on_party_conf**\ 执行
