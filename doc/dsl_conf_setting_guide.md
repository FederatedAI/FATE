## DSL & Task Submit Runtime Conf Setting

To make the modeling task more flexible, currently, FATE use its own domain-specific language(DSL) to describe modeling task. With usage of this DSL, modeling components such as data-io, feature-engineering and classification/regression module etc. can be combined as a Directed Acyclic Graph(DAG). Therefore, user can take and combine the algorithm components flexibly according to their needs.

In addition, each component has their own parameters to be configured. Also, the configuration may differ from party to party. For convenience, FATE use configure all parameters for all parties and all components in one file. This guide will show you how to create such a configure file.

### DSL Configure File

We use json file which is actually a dict as a dsl config file. The first level of the dict is always "components" showing you are going to add components in your modeling task.

    {
    "components" : {
            ...
        }
    }

Then each component should be defined in second level. Here is an example of setting a component:

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

As the example shows, user define the component name as key of this module. But this module should end up with a "_num" where the num should start with 0.

#### Field Specification
1. module: Specify which component use. This field should be strictly same with file name in federatedml/conf/setting_conf except the .json suffix.

2. input: There are two type of input, data and model.
    1. Data: There are three possible data_input type:
        1. data: typically used in data_io, feature_engineering modules and evaluation.
        2. train_data: Used in homo_lr, hetero_lr and secure_boost. If this field is provided, the task will be parse as a **fit** task
        3. eval_data: If train_data is provided, this field is optional. In this case, this data will be used as validation set. If train_data is not provided, this task will be parse as a **predict** or **transform** task.
    2. Model: There are two possible model-input type:
        1. model: This is a model input by same type of component. For example, hetero_binning_0 run as a fit component, and hetero_binning_1 take model output of hetero_binning_0 as input so that can be used to transform or predict.

        Here's an example showing this logic:

        ```
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
        ```

        2. isometric_model: This is used to specify the model input from upstream components.

        For example, feature selection will take feature binning as upstream model, since it will use information value as feature importance. Here's an example of feature selection component:
    ```
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

3. output: Same as input, two type of output may occur which are data and model.
    1. Data: Specify the output data name
    2. Model: Specify the output model name

    You can take the above case as an example.

### Submit Runtime Conf

Beside the dsl conf, users also need to prepare a submit runtime conf to set the parameters of each component.

1. initiator:
    To begin with, the initiator should be specified in this runtime conf. Here is an exmaple of setting initiator:
    ```
    "initiator": {
        "role": "guest",
        "party_id": 10000
    },
    ```

2. role:
    All the roles involved in this modeling task should be specified. Each element in the role should contain role name and their party ids. The reason for ids are with form of list is that there may exist multiple parties in one role.
    ```
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
    ```

3. role_parameters:
    Those parameters that are differ from party to party, should be indicated here. Please note that each parameters should has the form of list.

    1. Inside the role_parameters, party names are used as key and parameters of these parties are values. Take the following structure as an example:
    ```
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
    ```

    As this example shows, for each party, the input parameters such as train_data, eval_data and so on should be list in args. The name and namespace above are table indicators for uploaded data.

    Then, user can config parameters for each components. The component names should match names defined in the dsl config file. The content of each component parameters are defined in Param class located in federatedml/param.

4. algorithm_parameters:
    If there are some parameters are same among all parties, they can be set in algorithm_parameters. Here is an example showing how to do that.
    ```
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
    },
    ```
    Same with the form in role parameters, each key of the parameters are names of components that defined in dsl config file.

After finished these setting and submit the task, fate-flow will combine the parameters list in role-parameters and algorithm parameters. If there are still some fields are not defined, values in default runtime conf will be used. Then fate-flow will send these config files to their corresponding parties and start the federated modeling task.

### Multi-host configuration

For multi-host modeling case, all the host's party ids should be list in the role field.

```
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

Each parameter set for host should also be list in a list. The number of elements should match the number of hosts.

```
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
      },
```

The parameters set in algorithm parameters can keep it as it is. The parameters will be copied for every party.