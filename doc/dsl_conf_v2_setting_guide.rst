DSL & Task Submit Runtime Conf Setting V2
=========================================

[`中文`_]

.. _中文: dsl_conf_v2_setting_guide_zh.rst


To make the modeling task more flexible, currently, FATE uses its own domain-specific language(DSL)
to describe modeling task. With usage of this DSL, modeling components such as data-io,
feature-engineering and classification/regression module etc. can be combined as a Directed Acyclic Graph(DAG).
Therefore, user can take and combine the algorithm components flexibly according to their needs.

In addition, parameters of each component need to be configured.
Also, the configuration may vary from party to party.
For convenience, FATE configure all parameters for all parties and all components in one file.
This guide will show you how to create such a configure file.

Starting at FATE-1.5.0, V2 of dsl and submit conf is recommended, but user can still use old configuration method
of [`V1`_]

.. _V1: dsl_conf_v1_setting_guide.rst

DSL Configure File
------------------

1. Overview
~~~~~~~~~~~~
We use json file which is actually a dictionary as a dsl config file.

2. Components
~~~~~~~~~~~~~~

-  **definition:** components in your modeling task, always the first level of dsl dict.
-  **example:**

.. code-block:: json

  {
    "components" : {
            ...
        }
    }

-  **explanation:**

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

3. Module
~~~~~~~~~~~

-  **definition:** Specify which component to use.
-  **explanation:** This field should strictly match the file name in python/federatedml/conf/setting_conf except the ``.json`` suffix.
-  **example:**

.. code:: json

   "hetero_feature_binning_1": {
       "module": "HeteroFeatureBinning",
        ...
   }

4. Input
~~~~~~~~~~

-  **definition:** There are two types of input, data and model.

4.1 Data Input
^^^^^^^^^^^^^^^

-  **definition:**  Data input from previous modules; there are four possible data_input type:
   1. data: typically used in data_io, feature_engineering modules and evaluation.
   2. train_data: uses in training components like HeteroLR、HeteroSBT and so on. If this field is provided, the task will be parse as a **fit** task
   3. validate_data: If train_data is provided, this field is optional. In this case, this data will be used as validation set.
   4. test_data: specify the data used to predict, if this field is set up, the **model** also needs.

4.2 Model Input
^^^^^^^^^^^^^^^^^

-  **definition:**  Model input from previous modules; there are two possible model-input types:

1. model: This is a model input by the same type of component. For example, hetero_binning_0 run as a fit component, and hetero_binning_1 takes model output of hetero_binning_0 as input so that can be used to transform or predict.
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

2. isometric_model: This is used to specify the model input from upstream components.
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

4.3 Model Output
^^^^^^^^^^^^^^^^^

-  **definition:**  Same as input, two types of output may occur: which are data and model.


5.1 Data Output
^^^^^^^^^^^^^^^^^

-  **definition:** data output, there are four types:

1. data: normal data output
2. train_data: only for Data Split
3. validate_data: only for Data Split
4. test_data： only for Data Split

5.2 Model Output
^^^^^^^^^^^^^^^^^^
-  **definition:** model output, only use ``model``


JOB RUNTIME CONFIG Guide (for version 1.5.x and above)
-------------------------------------------------------

1. Overview
~~~~~~~~~~~~~~

Job Runtime Conf configures job and module settings for all
participants. Configurable values include:

2. DSL version
~~~~~~~~~~~~~~~~~~

-  **definition:** conf version, default 1, 2 is recommended
-  **example:**

.. code:: json

   "dsl_version": "2"

3. Job Participants
~~~~~~~~~~~~~~~~~~~~~~

3.1 Initiator
^^^^^^^^^^^^^^

-  **definition:** role and party_id of job initiator
-  **example:**

.. code:: json

   "initiator": {
       "role": "guest",
       "party_id": 9999
   }

3.2 Role
^^^^^^^^^^^

-  **definition:** Information on all participants
-  **explanation:** each key-value pair in ``role`` represents a role
   type and corresponding party ids; ``party_id`` should be specified as
   list since multiple parties may take the same role in a job
-  **examples**

.. code:: json

   "role": {
       "guest": [9999],
       "host": [10000],
       "arbiter": [10000]
   }

4. System Runtime Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **definition:** main system configuration when running jobs

4.1 Configuration Applicable Range Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``common``: applies to all participants
-  ``role``: applies only to specific participant; specify participant
   in :math:`role:`\ party_index format; note that ``role``
   configuration takes priority over ``common``

.. code:: json

   "common": {
   }

   "role": {
     "guest": {
       "0": {
       }
     }
   }

In the example above, configuration inside\ ``common`` applies to all
participants; configuration inside ``role-guest-0`` only applies to
participant ``guest_0``

Note: current version does not perform strict checking on role-specific
runtime parameters; ``common`` is suggested for setting runtime
configuration

4.2 Configurable Job Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     - 0, 1, 2
     - 0 for EGGROLL, 1 for SPARK with RabbitMQ, 2 for SPARK with Pulsar

   * - task_cores
     - 4
     - positive integer
     - total cpu cores requested

   * - task_parallelism
     - 1
     - positive int
     - maximum number of tasks allowed to run in parallel

   * - computing_partitions
     - same as task_cores
     - positive integer
     - partition number for table computing

   * - eggroll_run
     - \
     - processors_per_node
     - configuration specific for EGGROLL computing engine; generally set automatically based on ``task_cores``; if specified, ``task_cores`` value ineffective

   * - spark_run
     - \
     - num-executors, executor-cores
     - configuration specific for SPARK computing engine; generally set automatically based on ``task_cores``; if specified, ``task_cores`` value ineffective

   * - rabbitmq_run
     - \
     - queue, exchange etc.
     - parameters for rabbitmq to set up queue, exchange, etc.; generally takes system default

   * - federated_status_collect_type
     - PUSH
     - PUSH, PULL
     - way to collect federated job status; PUSH: participants report to initiator, PULL: initiator regularly queries from all participants

   * - timeout
     - 604800
     - positive int
     - time elapse (in second) for a job to timeout

   * - model_id
     - \-
     - \-
     - id of model, needed for prediction task

   * - model_version
     - \-
     - \-
     - version of model, needed for prediction task

.. note::

   1. Some types of ``computing_engine``, ``storage_engine``, and ``federation_engine``
   are only compatible with each other. For examples, SPARK
   ``computing_engine`` only supports HDFS ``storage_engine``.

   2. Combination of ``work_mode`` and ``backend`` automatically determines which
   three engines will be used.

   3. Developer may implement other types of engines and set new engine
   combinations in runtime conf.

4.3 Non-Configurable Job Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     - EGGROLL, RABBITMQ, STANDALONE, PULSAR
     - engine for communication among parties

   * - federated_mode
     - set automatically based on ``work_mode`` and ``backend``
     - SINGLE, MULTIPLE
     - federation mode

4.4 Example Job Parameter Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **EGGROLL** conf example with default CPU settings:

.. code-block:: json

     "job_parameters": {
        "common": {
           "work_mode": 1,
           "backend": 0,
           "task_cores": 4
        }
     }

2. **EGGROLL** conf example with manually specified CPU settings:

.. code-block:: json

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

3. **SPARK With RabbitMQ** conf example with manually specified CPU settings:

.. code-block:: json

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

4. **SPARK With Pulsar** conf example with default setting :

.. code-block:: json

     "job_parameters": {
        "common": {
            "job_type": "train",
            "work_mode": 1,
            "backend": 2,
            "spark_run": {
                "num-executors": 1,
                "executor-cores": 2
            }
        }
     }
4.5 Resource Management
^^^^^^^^^^^^^^^^^^^^^^^^^

Starting at version 1.5.0, FATE-Flow implements improved, more fine-grained resource management policy on cpu cores,
lifting restrictions on number of parallel tasks in previous versions.

4.5.1 Total Resource Setting
''''''''''''''''''''''''''''''''

-  resource comes from underlying engines; since current version does automatically obtain resource information from engines,
   FATE-Flow server obtains and register engine information to ``t_engine_registry`` from user-specified conf file \ ``$PROJECT_BASE/conf/service_conf.yaml``\
-  fate_on_eggroll：total_cores=cores_per_node*nodes
-  fate_on_spark：total_cores=cores_per_node*nodes
-  standalone：use \ **STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE**\ from \ ``$PROJECT_BASE/python/fate_flow/settings.py``\
-  separate computing resources for different engines
-  above settings effective after restarting FATE-Flow server

4.5.2 Calculate Computing Resource
''''''''''''''''''''''''''''''''''''

Calculate actual ``task_run_cores`` each task requests at computing engine, may not equal to the amount applied by resource manager

1. only set ``task_cores`` in job conf:

   -  task_run_cores(guest, host)：max(task_cores / total_nodes, 1) \* total_nodes
   -  task_run_cores(arbiter)：max(1 / total_nodes, 1) \* total_nodes
   -  FATE-Flow will automatically convert ``task_cores`` value into engine-specific configuration: eggroll.session.processors.per.node for EGGROLL, and executor-cores & num-executors for SPARK

2. set eggroll_run in job conf：

   -  task_run_cores(guest, host, arbiter)：eggroll.session.processors.per.node \* total_nodes

3. set spark_run in job conf：

   -  task_run_cores(guest, host, arbiter)：executor-cores \* num-executors

4.5.3 Resource Manager
'''''''''''''''''''''''''''''

1. Apply Resource for Jobs

   -  Computing Engine set to EGGROLL, STANDALONE

      -  apply_cores(guest, host): task_run_cores \* task_parallelism
      -  apply_cores(arbiter): 0, because actual resource cost is minimal and EGGROLL currently sets the same cores for all nodes, set to **0** to avoid unnecessary job queueing due to resource need from arbiter
      -  note: on EGGROLL cluster, each node always assigns arbiter task_run_cores/nodes cores

   -  Computing Engine set to SPARK

      -  SPARK supports executor-cores \* num-executors; not strongly correlated with number of cluster nodes due to SPARK own resource manager; if the calculated resource different from the one actually applied, jobs may keep waiting on SPARK engine
      -  apply_cores(guest, host, arbiter): task_run_cores \* task_parallelism

2. Job Management Policy

   -  Enqueue by job submission time
   -  Currently only support FIFO policy: manager only applies resources for the first job, deque the first job if success, wait for the next round if failure

3. Resource Application Policy

   -  Manager selects job following the above guidelines and distribute federated resource application request to all participants
   -  If all participants successfully secure resource, i.e.: (total_cores - apply_cores > 0), then the job succeeds in resource application
   -  If not all participants succeeds, then send rollback request to succeeded participants, and the job fails in resource application

5. Component Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5.1 Configuration Applicable Range Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``common``: applies to all participants
- ``role``: applies only to specific participant; specify participant in $role:$party_index format; note that ``role`` configuration takes priority over ``common``

.. code:: json

   "commom": {
   }

   "role": {
     "guest": {
       "0": {
       }
     }
   }


In the example above, configuration inside``common`` applies to all participants;
configuration inside ``role-guest-0`` only applies to participant `guest_0`

Note: current version now supports checking on both fields of specification.


5.2 Example Component Parameter Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Configuration of modules ``intersection_0``\ & \ ``hetero_lr_0``\ are put inside ``common``, thus applies to all participants
-  Configuration of modules \ ``reader_0``\ & \ ``dataio_0``\ are specified for each participant
-  Names of the above modules are specified in dsl file

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


5.3 Multi-host configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


5.4 Prediction configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5.4.1 Overview
'''''''''''''''''

Please note that in dsl v2，predict dsl is not automatically generated after training.
User should first deploy needed components with `Flow Client <../python/fate_client/flow_client/README.rst>`__.
Please refer to `FATE-Flow document <../python/fate_client/flow_client/README.rst#deploy>`__
for details on using deploy command:

.. code-block:: bash

    flow model deploy --model-id $model_id --model-version $model_version --cpn-list ...

Optionally, user can add additional component(s) to predict dsl, like ``Evaluation``:

5.4.2 Example
'''''''''''''''''

training dsl:

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

predict dsl:

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


6. Basic Workflow
~~~~~~~~~~~~~~~~~~~

1. After job submission, FATE-Flow obtains job dsl and job config and store them inside
   job folder under corresponding directory ``$PROJECT_BASE/jobs/$jobid/``
2. Parse job dsl & job config, generate fine-grained configuration according to provided settings
   (as mentioned above, backend & work_mode together determines configration for three engines) and fill
   in default parameter values
3. Distribute and store common configuration to each party, generate and store party-specific \ **job_runtime_on_party_conf**\ under jobs directory
4. Each party execute job following \ **job_runtime_on_party_conf**\
