# DSL & Task Submit Runtime Conf Setting V2

[[中文](dsl_conf_v2_setting_guide.zh.md)]

To make the modeling task more flexible, currently, FATE uses its own
domain-specific language(DSL) to describe modeling task. With usage of
this DSL, modeling components such as data-transform,
feature-engineering and classification/regression module etc. can be
combined as a Directed Acyclic Graph(DAG). Therefore, user can take and
combine the algorithm components flexibly according to their needs.

In addition, parameters of each component need to be configured. Also,
the configuration may vary from party to party. For convenience, FATE
configure all parameters for all parties and all components in one file.
This guide will show you how to create such a configure file.

Starting at FATE-1.5.0, V2 of dsl and submit conf is recommended.

## DSL Configure File

### 1. Overview

We use json file which is actually a dictionary as a dsl config file.

### 2. Components

  - **definition:** components in your modeling task, always the first
    level of dsl dict.
  - **example:**

    ```json
    {
      "components" : {
        ...
      }
    }
    ```

  - **explanation:**

    Then each component should be defined on the second level. Here is an example of setting a component:

    ```json
    "data_transform_0": {
          "module": "DataTransform",
          "input": {
              "data": {
                  "data": [
                      "reader_0.train_data"
                  ]
              }
          },
          "output": {
              "data": ["train"],
              "model": ["model"]
          }
      }
    ```

    As the example shows, user define the component name as key of this module.

    Please note that in DSL V2, all modeling task config should contain a **Reader** component to reader data from storage service, this component has "output" field only, like the following:

    ```json
    "reader_0": {
          "module": "Reader",
          "output": {
              "data": ["train"]
          }
    }
    ```

### 3. Module

  - **definition:** Specify which component to use.
  - **explanation:** This field should strictly match the ComponentMeta define in
    python file under the [fold](../../../python/federatedml/components)
  - **example:**


    ```json
    "hetero_feature_binning_1": {
        "module": "HeteroFeatureBinning",
        ...
    }
    ```

### 4. Input

  - **definition:** There are two types of input, data and model.

#### 4.1 Data Input

  - **definition:** Data input from previous modules; there are four
    possible data\_input type:
    1.  data: typically used in data\_transform, feature\_engineering modules
        and evaluation.
    2.  train\_data: uses in training components like HeteroLR、HeteroSBT
        and so on. If this field is provided, the task will be parse as
        a **fit** task
    3.  validate\_data: If train\_data is provided, this field is
        optional. In this case, this data will be used as validation
        set.
    4.  test\_data: specify the data used to predict, if this field is
        set up, the **model** also needs.

#### 4.2 Model Input

  - **definition:** Model input from previous modules; there are two
    possible model-input types:

    1.  model: This is a model input by the same type of component. For
        example, hetero\_binning\_0 run as a fit component, and
        hetero\_binning\_1 takes model output of hetero\_binning\_0 as input
        so that can be used to transform or predict. Here's an example
        showing this logic:
        
        ```json
        "hetero_feature_binning_1": {
            "module": "HeteroFeatureBinning",
            "input": {
                "data": {
                    "data": [
                        "data_transform_1.validate_data"
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
        ```

    2.  isometric\_model: This is used to specify the model input from
        upstream components. For example, feature selection will take
        feature binning as upstream model, since it will use information
        value as feature importance. Here's an example of feature selection
        component:
        
        ```json
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
        ```

#### 4.3 Model Output

  - **definition:** Same as input, two types of output may occur: which
    are data and model.

#### 5.1 Data Output

  - **definition:** data output, there are four types:

    1.  data: normal data output
    2.  train\_data: only for Data Split
    3.  validate\_data: only for Data Split
    4.  test\_data：only for Data Split

#### 5.2 Model Output

  - **definition:** model output, only use `model`

## JOB RUNTIME CONFIG Guide (for version 1.5.x and above)

### 1. Overview

Job Runtime Conf configures job and module settings for all
participants. Configurable values include:

### 2. DSL version

  - **definition:** conf version, default 1, 2 must be set if fate's version >= 1.7
  - **example:**

    ```json
    "dsl_version": 2
    ```

### 3. Job Participants

#### 3.1 Initiator

  - **definition:** role and party\_id of job initiator
  - **example:**

    ```json
    "initiator": {
        "role": "guest",
        "party_id": 9999
    }
    ```

#### 3.2 Role

  - **definition:** Information on all participants
  - **explanation:** each key-value pair in `role` represents a role
    type and corresponding party ids; `party_id` should be specified as
    list since multiple parties may take the same role in a job
  - **examples**

    ```json
    "role": {
        "guest": [9999],
        "host": [10000],
        "arbiter": [10000]
    }
    ```

### 4. System Runtime Parameters

  - **definition:** main system configuration when running jobs

#### 4.1 Configuration Applicable Range Policy

  - `common`: applies to all participants
  - `role`: applies only to specific participant; specify participant in
    \(role:\)party\_index format; note that `role` configuration takes
    priority over `common`

    ```json
    "common": {
    }

    "role": {
      "guest": {
        "0": {
        }
      }
    }
    ```

In the example above, configuration inside`common` applies to all
participants; configuration inside `role-guest-0` only applies to
participant `guest_0`

Note: current version does not perform strict checking on role-specific
runtime parameters; `common` is suggested for setting runtime
configuration

#### 4.2 Configurable Job Parameters

| Parameter Name                   | Default Value       | Acceptable Values             | Information                                                                                                                                          |
| -------------------------------- | ------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| inheritance\_info                |                     | job_id, component_list        | jobid and component list of job that need to be inherited                                                                                                                                             |
| job\_type                        | train               | train, predict                | job type                                                                                                                                             |
| task\_cores                      | 4                   | positive integer              | total cpu cores requested                                                                                                                            |
| task\_parallelism                | 1                   | positive int                  | maximum number of tasks allowed to run in parallel                                                                                                   |
| computing\_partitions            | same as task\_cores | positive integer              | partition number for table computing                                                                                                                 |
| eggroll\_run                     |                     | processors\_per\_node         | configuration specific for EGGROLL computing engine; generally set automatically based on `task_cores`; if specified, `task_cores` value ineffective |
| spark\_run                       |                     | num-executors, executor-cores | configuration specific for SPARK computing engine; generally set automatically based on `task_cores`; if specified, `task_cores` value ineffective   |
| rabbitmq\_run                    |                     | queue, exchange etc.          | parameters for rabbitmq to set up queue, exchange, etc.; generally takes system default                                                              |
| federated\_status\_collect\_type | PUSH                | PUSH, PULL                    | way to collect federated job status; PUSH: participants report to initiator, PULL: initiator regularly queries from all participants                 |
| timeout                          | 259200 (3 days)     | positive int                  | time elapse (in second) for a job to timeout                                                                                                         |
| model\_id                        | \-                  | \-                            | id of model, needed for prediction task                                                                                                              |
| model\_version                   | \-                  | \-                            | version of model, needed for prediction task                                                                                                         |

Configurable Job Parameters

!!! Note

    1. Some types of `computing_engine`, `storage_engine` are only compatible with each other. 
    2. Developer may implement other types of engines and set new engine combinations in runtime conf.

#### 4.3 Non-Configurable Job Parameters

| Parameter Name     | Default Value                                        | Acceptable Values                     | Information                            |
| ------------------ | ---------------------------------------------------- | ------------------------------------- | -------------------------------------- |
| computing\_engine  | Configure in service_conf.yaml | EGGROLL, SPARK, STANDALONE            | engine for computation                 |
| storage\_engine    | Configure in service_conf.yaml | EGGROLL, HDFS, STANDALONE             | engine for storage                     |
| federation\_engine | Configure in service_conf.yaml | EGGROLL, RABBITMQ, STANDALONE, PULSAR | engine for communication among parties |
| federated\_mode    | set automatically based on `federation_engine` | SINGLE, MULTIPLE                      | federation mode                        |

Non-configurable Job Parameters

#### 4.4 Example Job Parameter Configuration

1.  **EGGROLL** conf example with default CPU settings:

    ```json
    "job_parameters": {
      "common": {
          "task_cores": 4
      }
    }
    ```

2.  **EGGROLL** conf example with manually specified CPU settings:

    ```json
    "job_parameters": {
      "common": {
          "job_type": "train",
          "eggroll_run": {
            "eggroll.session.processors.per.node": 2
          },
          "task_parallelism": 2,
          "computing_partitions": 8,
          "timeout": 36000,
      }
    }
    ```

3.  **SPARK With RabbitMQ** conf example with manually specified CPU
    settings:

    ```json
    "job_parameters": {
      "common": {
          "job_type": "train",
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
    ```

4.  **SPARK With Pulsar** conf example with default setting :

    ```json
    "job_parameters": {
      "common": {
          "job_type": "train",
          "spark_run": {
              "num-executors": 1,
              "executor-cores": 2
          }
      }
    }
    ```

#### 4.5 Resource Management

Starting at version 1.5.0, FATE-Flow implements improved, more
fine-grained resource management policy on cpu cores, lifting
restrictions on number of parallel tasks in previous versions.

##### 4.5.1 Total Resource Setting

  - resource comes from underlying engines; since current version does
    automatically obtain resource information from engines, FATE-Flow
    server obtains and register engine information to
    `t_engine_registry` from user-specified conf file
    `$PROJECT_BASE/conf/service_conf.yaml`
  - fate\_on\_eggroll：total\_cores=cores\_per\_node\*nodes
  - fate\_on\_spark：total\_cores=cores\_per\_node\*nodes
  - fate\_on\_standalone：total\_cores=cores\_per\_node\*nodes
  - separate computing resources for different engines
  - above settings effective after restarting FATE-Flow server

##### 4.5.2 Calculate Computing Resource

Calculate actual `task_run_cores` each task requests at computing
engine, may not equal to the amount applied by resource manager

1.  only set `task_cores` in job conf:

      - task\_run\_cores(guest, host)：max(task\_cores / total\_nodes, 1)
        \* total\_nodes
      - task\_run\_cores(arbiter)：max(1 / total\_nodes, 1) \*
        total\_nodes
      - FATE-Flow will automatically convert `task_cores` value into
        engine-specific configuration:
        eggroll.session.processors.per.node for EGGROLL, and
        executor-cores & num-executors for SPARK
2.  set eggroll\_run in job conf：

      - task\_run\_cores(guest, host,
        arbiter)：eggroll.session.processors.per.node \* total\_nodes
3.  set spark\_run in job conf：

      - task\_run\_cores(guest, host, arbiter)：executor-cores \*
        num-executors

##### 4.5.3 Resource Manager

1.  Apply Resource for Jobs
      - Computing Engine set to EGGROLL, STANDALONE
          - apply\_cores(guest, host): task\_run\_cores \*
            task\_parallelism
          - apply\_cores(arbiter): 0, because actual resource cost is
            minimal and EGGROLL currently sets the same cores for all
            nodes, set to **0** to avoid unnecessary job queueing due to
            resource need from arbiter
          - note: on EGGROLL cluster, each node always assigns arbiter
            task\_run\_cores/nodes cores
      - Computing Engine set to SPARK
          - SPARK supports executor-cores \* num-executors; not strongly
            correlated with number of cluster nodes due to SPARK own
            resource manager; if the calculated resource different from
            the one actually applied, jobs may keep waiting on SPARK
            engine
          - apply\_cores(guest, host, arbiter): task\_run\_cores \*
            task\_parallelism
2.  Job Management Policy
      - Enqueue by job submission time
      - Currently only support FIFO policy: manager only applies
        resources for the first job, deque the first job if success,
        wait for the next round if failure
3.  Resource Application Policy
      - Manager selects job following the above guidelines and
        distribute federated resource application request to all
        participants
      - If all participants successfully secure resource, i.e.:
        (total\_cores - apply\_cores \> 0), then the job succeeds in
        resource application
      - If not all participants succeeds, then send rollback request to
        succeeded participants, and the job fails in resource
        application

### 5. Component Parameter Configuration

#### 5.1 Configuration Applicable Range Policy

  - `common`: applies to all participants
  - `role`: applies only to specific participant; specify participant in
    $role:$party\_index format; note that `role` configuration takes
    priority over `common`

```json
"commom": {
}

"role": {
  "guest": {
    "0": {
    }
  }
}
```

In the example above, configuration inside`common` applies to all
participants; configuration inside `role-guest-0` only applies to
participant `guest_0`

!!!Note
    current version now supports checking on both fields of specification.

#### 5.2 Example Component Parameter Configuration

  - Configuration of modules `intersection_0`& `hetero_lr_0`are put
    inside `common`, thus applies to all participants
  - Configuration of modules `reader_0`& `data_transform_0`are specified
    for each participant
  - Names of the above modules are specified in dsl file

```json
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
        "data_transform_0":{
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
        "data_transform_0":{
          "with_label": false,
          "output_format": "dense"
        }
      }
    }
  }
}
```

#### 5.3 Multi-host configuration

For multi-host modeling case, all the host's party ids should be list in
the role field.

```json
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
```

Each parameter set for host should also be config The number of elements
should match the number of hosts.

```json
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
```

The parameters set in common parameters need not be copied into host
role parameters. Common parameters will be copied for every party.

#### 5.4 Prediction configuration

##### 5.4.1 Overview

Please note that in dsl v2，predict dsl is not automatically generated
after training. User should first deploy needed components with [Flow
Client](https://fate-flow.readthedocs.io/en/latest/fate_flow_client/). Please refer to
[FATE-Flow
document](https://github.com/FederatedAI/FATE-Flow/blob/main/doc/cli/model.md#deploy) for
details on using deploy
command:

```bash
$ flow model deploy --model-id $model_id --model-version $model_version --cpn-list ...
```

Optionally, user can add additional component(s) to predict dsl, like
`Evaluation`:

##### 5.4.2 Example

training dsl:

```json
"components": {
    "reader_0": {
        "module": "Reader",
        "output": {
            "data": [
                "data"
            ]
        }
    },
    "data_transform_0": {
        "module": "DataTransform",
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
                    "data_transform_0.data"
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
```

predict dsl:

```json
"components": {
    "reader_0": {
        "module": "Reader",
        "output": {
            "data": [
                "data"
            ]
        }
    },
    "data_transform_0": {
        "module": "DataTransform",
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
                    "data_transform_0.data"
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
```

### 6. Basic Workflow

1.  After job submission, FATE-Flow obtains job dsl and job config and
    store them inside job folder under corresponding directory
    `$PROJECT_BASE/fateflow/jobs/$jobid/`
2.  Parse job dsl & job config, generate fine-grained configuration
    according to provided settings and
    fill in default parameter values
3.  Distribute and store common configuration to each party, generate
    and store party-specific **job\_runtime\_on\_party\_conf**under jobs
    directory
4.  Each party execute job following **job\_runtime\_on\_party\_conf**
