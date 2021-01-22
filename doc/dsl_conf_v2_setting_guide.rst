DSL & Task Submit Runtime Conf Setting V2
======================================

To make the modeling task more flexible, currently, FATE uses its own domain-specific language(DSL)
to describe modeling task. With usage of this DSL, modeling components such as data-io,
feature-engineering and classification/regression module etc. can be combined as a Directed Acyclic Graph(DAG).
Therefore, user can take and combine the algorithm components flexibly according to their needs.

In addition, parameters of each component need to be configured.
Also, the configuration may vary from party to party.
For convenience, FATE configure all parameters for all parties and all components in one file.
This guide will show you how to create such a configure file.

In FATE's version since 1.5.0, V2 of dsl and submit conf will be recommend, but user can still use old configuration method
of [`V1`_]

.. _V1: dsl_conf_v1_setting_guide.rst

Please note that dsl V2 will not support online serving in fate-1.5.0，it will be support in later version.

DSL Configure File
------------------

We use json file which is actually a dict as a dsl config file. The first level of the dict is always "components," which indicates content in the dict are components in your modeling task.

.. code-block:: json
  
  {
    "components" : {
            ...
        }
    }


Then each component should be defined on the second level. Here is an example of setting a component:

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


As the example shows, user define the component name as key of this module.

Please note that in DSL V2, all modeling task config should contain a **Reader** component to reader data from storage service,
this component has "output" field only, like the following:

.. code-block:: json

  "reader_0": {
        "module": "Reader",
        "output": {
            "data": ["train"]
        }
  }

Field Specification
^^^^^^^^^^^^^^^^^^^

:module: Specify which component to use. This field should strictly match the file name in python/federatedml/conf/setting_conf except the ``.json`` suffix.

:input: There are two types of input, data and model.

    - Data: There are four possible data_input type:

      1. data: typically used in data_io, feature_engineering modules and evaluation.
      2. train_data: uses in training components like HeteroLR、HeteroSBT and so on. If this field is provided, the task will be parse as a **fit** task
      3. validate_data: If train_data is provided, this field is optional. In this case, this data will be used as validation set.
      4. test_data: specify the data used to predict, if this field is set up, the **model** also needs.

    - Model: There are two possible model-input types:

      - model: This is a model input by the same type of component. For example, hetero_binning_0 run as a fit component, and hetero_binning_1 takes model output of hetero_binning_0 as input so that can be used to transform or predict.
        Here's an example showing this logic:

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

      - isometric_model: This is used to specify the model input from upstream components.
        For example, feature selection will take feature binning as upstream model, since it will use information value as feature importance. Here's an example of feature selection component:

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

:output: Same as input, two types of output may occur which are data and model.
    
    1. Data: Specify the output data name
    2. Model: Specify the output model name

    You can take the above case as an example.


Submit Runtime Conf
-------------------

Besides the dsl conf, user also need to prepare a submit runtime conf to set parameters for each component.

:dsl_version:
  To enabled using of dsl V2, this field should be set.

  .. code-block:: json

     "dsl_version": 2

:initiator:
  To begin with, the initiator should be specified in this runtime conf. Here is an example of setting initiator:

  .. code-block:: json

     "initiator": {
        "role": "guest",
        "party_id": 10000
     }


:role:
  All the roles involved in this modeling task should be specified. Each role comes with role name and corresponding party id(s).
  Ids are always specified in the form of list since there may exist multiple parties of the same role.

  .. code-block:: json

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

:component_parameters:
  Running parameters for components included in dsl should be specified here.

  It contains two sub-fields ``common`` and ``role``:

  * parameter specification under ``common`` field applies to all parties
  * parameter values under ``role`` field are only taken by each corresponding party

  .. code-block:: json

     "component_parameters": {
         "common": {
             "component_x": {
                 ...
             },
             ...
         },
         "role": {
             ...
         }
     }

  :role:
    Inside the ``role`` field, party names are used as key, parameter specification as values.

    Take the following json as an example:

    .. code-block:: json

       "role": {
            "guest": {
                "0": {
                    "reader_0": {
                        "table": {
                                    "namespace": "guest",
                                    "name": "table"
                        }
                    },
                    "dataio_0": {
                        "input_format": "dense",
                        "with_label": true
                    }
                }
            },
            "host": {
                "0": {
                    "reader_0": {
                        "table": {
                                    "namespace": "host",
                                    "name": "table"}
                        },
                    "dataio_0": {
                        "input_format": "tag",
                        "with_label": false
                    }
                }
            }
        }

    "0" indicates that it is the 0_th party of some role(indexing starts at 0).

    User can config parameters for each component.

    Component names should match those defined in the dsl config file.

    Parameters of each component are defined in `Param <../python/federatedml/param>`_ class.

    Parties can be packed together and share configuration, for example:

    .. code-block:: json

       "role": {
            "host": {
                "0|2": {
                    "dataio_0": {
                        "input_format": "tag",
                        "with_label": false
                    }
                },
                "1": {
                    "dataio_0": {
                        "input_format": "dense",
                        "with_label": false
                    }
                }
            }
        }

  :common:
    If some parameters are the same among all parties, they can be set in ``common``. Here is an example:

    .. code-block:: json

        "common": {
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
        }

    Same ``role``, keys are the names of components defined in dsl config file and values parameter configuration.

:job_parameters:
  Please note that to enable DSL V2, **dsl_version** must be set to **2**.

  Same as component_parameters, it also has two sub-fields ``common`` and ``role``:

  * parameter specification under ``common`` field applies to all parties
  * parameter values under ``role`` field are only taken by each corresponding party

  .. code-block:: json

     "job_parameters": {
          "common": {
             ...
          },
          "role": {
             ...
          }
     }

.. list-table:: Configurable Job Parameters
   :widths: 20 20 30 30
   :header-rows: 1

   * - Parameter Name
     - Default Value
     - Acceptable Values
     - Information

   * - job_type
     - train
     - train, predict
     - job type

   * - work_mode
     - 0
     - 0, 1
     - 0 for standalone, 1 for cluster

   * - backend
     - 0
     - 0, 1
     - 0 for EGGROLL, 1 for SPARK

   * - federated_status_collect_type
     - PUSH
     - PUSH, PULL
     - type of collecting job status

   * - timeout
     - 604800
     - positive int
     - time elapse (in second) for a job to timeout

   * - eggroll_run
     -
     - most commonly used is "eggroll.session.processors.per.node", details can be found in `EggRoll configuration  <https://github.com/WeBankFinTech/eggroll/wiki/eggroll.properties:-Eggroll's-Main-Configuration-File>`_.
     - parameter for EGGROLL computing engine

   * - spark_run
     -
     - num-executors, executor-cores
     - parameter for SPARK computing engine

   * - rabbitmq_run
     -
     - queue, exchange etc.
     - parameters for creation of queue, exchange in rabbitmq

   * - task_parallelism
     - 2
     - positive int
     - maximum number of tasks allowed to run in parallel

   * - model_id
     - \-
     - \-
     - if of model, needed for prediction task

   * - model_version
     - \-
     - \-
     - version of model, needed for prediction task

.. list-table:: Non-configurable Job Parameters
   :widths: 20 20 30 30
   :header-rows: 1

   * - Parameter Name
     - Default Value
     - Acceptable Values
     - Information

   * - computing_engine
     - set automatically based on ``work_mode`` and ``backend``
     - EGGROLL, SPARK, STANDALONE
     - engine for computation

   * - storage_engine
     - set automatically based on ``work_mode`` and ``backend``
     - EGGROLL, HDFS, STANDALONE
     - engine for storage

   * - federation_engine
     - set automatically based on ``work_mode`` and ``backend``
     - EGGROLL, RABBITMQ, STANDALONE
     - engine for communication among parties

   * - federated_mode
     - set automatically based on ``work_mode`` and ``backend``
     - SINGLE, MULTIPLE
     - federation mode

.. note::

   1. Some types of ``computing_engine``, ``storage_engine``, and ``federation_engine``
   are only compatible with each other. For examples, SPARK
   ``computing_engine`` only supports HDFS ``storage_engine``.

   2. Combination of ``work_mode`` and ``backend`` automatically determines which
   combination of engines will be used.

   3. Developer may implement other types of engines and set new engine
   combinations.

**EGGROLL** conf example:

.. code-block:: json

     "job_parameters": {
        "common": {
           "work_mode": 1,
           "backend": 0,
           "eggroll_run": {
              "eggroll.session.processors.per.node": 2
           }
        }
     }

**SPARK** conf example:

.. code-block:: json

     "job_parameters": {
        "common": {
            "work_mode": 1,
            "backend": 1,
            "spark_run": {
               "num-executors": 1,
               "executor-cores": 2
            }
        }
     }

After setting config files and submitting the task, fate-flow will combine the parameter list in role-parameters and algorithm parameters.
If there are still some undefined fields, default parameter values will be used.
FATE Flow will send these config files to their corresponding parties and start federated task.


Multi-host configuration
------------------------

For multi-host modeling case, all the host's party ids should be list in the role field.

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

Each parameter set for host should also be config The number of elements should match the number of hosts.

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

The parameters set in common parameters need not be copied into host role parameters.
Common parameters will be copied for every party.


Prediction configuration
------------------------

Please note that in dsl v2，predict dsl is nnot automatically generated after training.
User should first deploy needed components.
Please refer to`FATE-Flow CLI <../python/fate_flow/doc/Fate_Flow_CLI_v2_Guide.rst#dsl>`__'
for details on using deploy command:

.. code-block:: bash

    flow job dsl --cpn-list ...

**Examples**
Use a training dsl:

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

Use the following command to generate predict dsl:

.. code-block:: bash

    flow job dsl --train-dsl-path $job_dsl --cpn-list "reader_0, dataio_0, intersection_0, hetero_nn_0" --version 2 -o ./

Generated dsl:

.. code-block::: json

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
                "model": [
                    "pipeline.dataio_0.model"
                ],
                "data": {
                    "data": [
                        "reader_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ]
            }
        },
        "intersection_0": {
            "module": "Intersection",
            "output": {
                "data": [
                    "data"
                ]
            },
            "input": {
                "data": {
                    "data": [
                        "dataio_0.data"
                    ]
                }
            }
        },
        "hetero_nn_0": {
            "module": "HeteroNN",
            "input": {
                "model": [
                    "pipeline.hetero_nn_0.model"
                ],
                "data": {
                    "test_data": [
                        "intersection_0.data"
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

Optionally, use can add additional component(s) to predict dsl, like ``Evaluation``:

.. code-block:: json

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
