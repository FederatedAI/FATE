DSL 配置和运行配置 V1
====================
[`ENG`_]

.. _ENG: dsl_conf_v1_setting_guide.rst

为了让任务模型的构建更加灵活，目前 FATE 使用了一套自定的领域特定语言 (DSL) 来描述任务。在 DSL 中，各种模块（例如数据读写 data_io，特征工程 feature-engineering， 回归 regression，分类 classification）可以通向一个有向无环图 （DAG） 组织起来。通过各种方式，用户可以根据自身的需要，灵活地组合各种算法模块。

除此之外，每个模块都有不同的参数需要配置，不同的 party 对于同一个模块的参数也可能有所区别。为了简化这种情况，对于每一个模块，FATE 会将所有 party 的不同参数保存到同一个运行配置文件（Submit Runtime Conf）中，并且所有的 party 都将共用这个配置文件。这个指南将会告诉你如何创建一个 DSL 配置文件。

DSL 配置文件
-------------

DSL 的配置文件采用 json 格式，实际上，整个配置文件就是一个 json 对象 （dict）。在这个 dict 的第一级是 "components"，用来表示这个任务将会使用到的各个模块。

::
  
  {
    "components" : {
            ...
        }
    }


每个独立的模块定义在 "components" 之下，例如：

::
  
  "dataio_0": {
        "module": "DataIO",
        "input": {
            "data": {
                "data": [
                    "args.train_data"
                ]
            }
        },
        "output": {
            "data": ["train"],
            "model": ["dataio"]
        },
        "need_deploy": true
    }


正如这个例子，用户需要使用模块名加数字 `\_num` 作为对应模块的 key，例如 `dataio_0`，并且数字应从 0 开始计数。

参数说明
^^^^^^^^^^^

:module:
   用来指定使用的模块。这个参数的内容需要和 `federatedml/conf/setting_conf` 下各个模块的文件名保持一致（不包括 .json 后缀）。

:input:
   分为两种输入类型，分别是 data 和 model。

   1. Data: 有三种可能的输入类型

      1. data: 一般被用于 data_io 模块, feature_engineering 模块或者 evaluation 模块
      2. train_data: 一般被用于 homo_lr, heero_lr 和 secure_boost 模块。如果出现了 train_data 字段，那么这个任务将会被识别为一个 fit 任务
      3. eval_data: 如果存在 train_data 字段，那么该字段是可选的。如果选择保留该字段，则 eval_data 指向的数据将会作为 validation set。若不存在 train_data 字段，则这个任务将被视作为一个 predict 或 transform 任务。 

   2. Model: 有两种可能的输入类型：

      1. model: 用于同种类型组件的模型输入。例如，hetero_binning_0 会对模型进行 fit，然后 hetero_binning_1 将会使用 hetero_binning_0 的输出用于 predict 或 transform。代码示例：
         ::

            "hetero_feature_binning_1": {
              "module": "HeteroFeatureBinning",
              "input": {
                "data": {
                  "data": [
                    "dataio_1.eval_data"
                  ]
                },
                "model": [
                  "hetero_feature_binning_0.fit_model"
                ]
              },
              "output": {
                "data": ["eval_data"],
                "model": ["eval_model"]
              }
            }

      2. isometric_model: 用于指定继承上游组件的模型输入。 例如，feature selection 的上游组件是 feature binning，它将会用到 feature binning 的信息来作为 feature importance。代码示例：
 
         ::

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

   3. output: 和 input 一样，有 data 和 model 两种类型。
      
      1. Data: 指定输出的 data 名
      2. Model: 指定输出的 model 名


运行配置 Submit Runtime Conf
----------------------------

除了 DSL 的配置文件之外，用户还需要准备一份运行配置（Submit Runtime Conf）用于设置各个组件的参数。

:initiator:
   在运行配置的开头，用户需要定义 initiator。例如
   ::

      "initiator": {
        "role": "guest",
        "party_id": 10000
      }

:role:
   所有参与这个任务的 roles 都需要在运行配置中指定。在 role 字段中，每一个元素代表一种角色以及承担这个角色的 party_id。每个角色的 party_id 以列表形式存在，因为一个任务可能涉及到多个 party 担任同一种角色。
   ::
    
       "role": {
         "guest": [
           10000
         ],
         "host": [
           10000
         ],
         "arbiter": [
           10000
         ]
       }

:role_parameters:
   这一部分的参数对于不同的 party 都有所区别。同样地，每一个参数也是用列表的方式呈现。在 role_parameters 中，party 名被作为每一项元素的 key，而 value 则是具体提的参数内容。例如：
   ::
    
       "guest": {
          "args": {
            "data": {
              "train_data": [
                {
                  "name": "1ca0d9eea77e11e9a84f5254005e961b",
                  "namespace": "arbiter-10000#guest-10000#host-10000#train_input#guest#10000"
                }
              ]
            }
          },
          "dataio_0": {
            "with_label": [
              true
            ],
            ...
          }
        },
        "host": {
          "args": {
            "data": {
              "train_data": [
                {
                  "name": "3de22bdaa77e11e99c5d5254005e961b",
                  "namespace": "arbiter-10000#guest-10000#host-10000#train_input#host#10000"
                }
              ]
            }
          },
          "dataio_0": {
             ...
          }
          ...
        }
    
    
   就像上面这个例子，对于每一个 party，它们的输入参数 train_data，eval_data 都应该以列表形式存在。name 和 namespace 字段则是用来指定用来上传数据的表格位置。

   用户还可以分别配置每一个组件的参数。组件名需要和 DSL 配置文件中的组件名保持一致。每个组件具体的参数列表可以在 `federatedml/param` 的 `Param` 类中找到。

:algorithm_parameters:
   如果用户希望定义一些所有 party 都共享的参数，那么可以在 algorithm_parameters 中设置。例如：

   ::

       "hetero_feature_binning_0": {
         ...
       },
       "hetero_feature_selection_0": {
            ...
       },
       "hetero_lr_0": {
         "penalty": "L2",
         "optimizer": "rmsprop",
         "eps": 1e-5,
         "alpha": 0.01,
         "max_iter": 10,
         "converge_func": "diff",
         "batch_size": 320,
           "learning_rate": 0.15,
         "init_param": {
           "init_method": "random_uniform"
         },
         "cv_param": {
           "n_splits": 5,
           "shuffle": false,
           "random_seed": 103,
           "need_cv": false,
          }
        }

   和上一个部分一样，在 algorithm_parameters 中，每一个参数的 key 都是在 DSL 配置文件中定义好的组件名。

在完成这些配置文件并提交任务之后，fate-flow 将会把 role_parameters 和 algorithm_parameters 中的所有参数合并。如果合并之后，仍然存在没有定义的参数，fate-flow 则会使用默认值。fate-flow 会将这些参数分发到对应的 party，并开始联邦建模任务。

多个 Host 情况下的配置
-----------------------

对于存在多个 Host 的模型，所有 Host 的 party_id 都应该在 role 中列举出来。例如：
::

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


每个针对 Host 的参数都应该以列表的方式储存，列表中组件的个数和 Host 的个数应保持一致。
::

   "host": {
     "args": {
       "data": {
         "train_data": [
           {
             "name": "hetero_breast_host_1",
             "namespace": "hetero_breast_host"
           },
           {
             "name": "hetero_breast_host_2",
             "namespace": "hetero_breast_host"
           },
           {
             "name": "hetero_breast_host_3",
             "namespace": "hetero_breast_host"
           }
         ]
       }
     },
     "dataio_0": {
     "with_label": [false, false, false],
     "output_format": ["dense", "dense", "dense"],
     "outlier_replace": [true, true, true]
   }

注意 algorithm_parameters 里面的参数不需要额外处理，FATE 会自动把这些参数复制给每一个 party。
