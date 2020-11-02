# Model Migrate Guide

The model migration function is mainly to solve the problem that model files are not available after being transferred between different machines in different clusters.

Let say we have trained a model in Party A and B and wish to use it to predict in Party C and D. Model Migration would be a tool to achieve it. 

The migration process can be listed as follow:

## 1. Edit Configuration File

Edit a configuration file which setting the migration information in original cluster (or machines). An example configuration file can be referred to [migrate_model.json](https://github.com/FederatedAI/FATE/blob/master/python/fate_flow/examples/migrate_model.json)

```json
{
  "job_parameters": {
    "federated_mode": "MULTIPLE"
  },
  "migrate_initiator": {
    "role": "guest",
    "party_id": 99
  },
  "role": {
    "guest": [9999],
    "arbiter": [10000],
    "host": [10000]
  },
  "migrate_role": {
    "guest": [99],
    "arbiter": [100],
    "host": [100]
  },
  "execute_party": {
    "guest": [9999],
    "arbiter": [10000],
    "host": [10000]
  },
  "model_id": "arbiter-10000#guest-9999#host-10000#model",
  "model_version": "202006171904247702041",
  "unify_model_version": "20200901_0001"
}
```

The explanation of these parameters are:

1. **`job_parameters`**：Should be one of the two strings: `MULTIPLE` and `SINGLE`. If set as `MULTIPLE`, the initiator (typically the guest) will send and execute this migration task to the `execute_party` list below. Otherwise, this task will be executed in the initiator only. 
2. **`migrate_initiator`**：Indicate the initiator of the migrated model. `role` and `party_id` are needed here.
3. **`role`**：Indicate the `role` and `party_id` information of the cluster which the original model generated from. 
4. **`migrate_role`**：Indicate the `role` and `party_id` information of the cluster which to be migrate to.
5. **`execute_party`**：Used when `job_parameters` is set as `MULTIPLE`. Indicate the `role` and `party_id` that need to execute migration.
6. **`model_id`**：The `model_id` of original model. It could be obtained in the return message when starting a task. 
7. **`model_version`**：The `model_version` of original model. It could be obtained in the return message when starting a task. 
8. **`unify_model_version`**：Optional. If set, this string will be used as `model_version` of the migrated model. Otherwise, the `job_id` ot this migration task will be used as its `model_version`. 



## 2. Submit Migration Task

As mentioned above, if a federated migration(multiple parties migrate simultaneously) is required, the **`job_parameters`** should be set as `MULTIPLE`. Otherwise, if the migration happens on the local side(guest, host and arbiter are all available) only, the **`job_parameters`** should be set as `SINGLE`

Use the following FATE Flow CLI v2 command to submit this execution:

```bash
flow model migrate -c /data/projects/fate/python/fate_flow/examples/migrate_model.json
```



## 3. Task Execution Result

Take the following configuration as an example：

```json
{
  "job_parameters": {
    "federated_mode": "MULTIPLE"
  },
  "migrate_initiator": {
    "role": "guest",
    "party_id": 9999
  },
  "role": {
    "guest": [9999],
    "host": [10000]
  },
  "migrate_role": {
    "guest": [99],
    "host": [100]
  },
  "execute_party": {
    "guest": [9999],
    "host": [10000]
  },
  "model_id": "guest-9999#host-10000#model",
  "model_version": "202010291539339602784",
  "unify_model_version": "fate_migration"
}
```

This configuration file shows the following information.

* The original model is trained by the cluster in which the party id of guest is 9999 and host is 10000. 
* The model_id and model_version of this model are `guest-9999#host-10000#model` and `202010291539339602784` respectively. 
* We take another cluster, in which the guest's party_id is 99 and host's is 100, as a target migration cluster. 
* We have set the model_version of migrated model as `fate_migration`. 
* After submitting, this task will be executed in guest(9999) and host(10000) simultaneously. 

If success, the expected return result is:

```json
{
    "data": {
        "detail": {
            "guest": {
                "9999": {
                    "retcode": 0,
                    "retmsg": "Migrating model successfully. The configuration of model has been modified automatically. New model id is: guest-99#host-100#model, model version is: fate_migration. Model files can be found at '/data/projects/fate/temp/fate_flow/guest#99#guest-99#host-100#model_fate_migration.zip'."
                }
            },
            "host": {
                "10000": {
                    "retcode": 0,
                    "retmsg": "Migrating model successfully. The configuration of model has been modified automatically. New model id is: guest-99#host-100#model, model version is: fate_migration. Model files can be found at '/data/projects/fate/temp/fate_flow/host#100#guest-99#host-100#model_fate_migration.zip'."
                }
            }
        },
        "guest": {
            "9999": 0
        },
        "host": {
            "10000": 0
        }
    },
    "jobId": "202010292152299793981",
    "retcode": 0,
    "retmsg": "success"
}
```

After the task is successfully executed, a compressed file of the migrated model will be generated on the executor's machine. The file path can be obtained in the return result. For instances, the compressed file path in guest is: /data/projects/fate/temp/fate_flow/guest#99#guest-99#host-100#model_fate_migration.zip and in host is: /data/projects/fate/temp/fate_flow/host#100#guest-99#host-100#model_fate_migration.zip. New model_id and model_version of the migrated model can be obtained in this return message too. 


## 4. Transfer and Import Files

After the migration task executed successful, the compressed files are needed to be transferred to the corresponding machine manually. Take the example shown above, the compressed file in guest(9999) should be transferred to guest(99). This file can be located in any path you like as long as you indicate it in the following import configuration. An example of this import configuration file is [import_model.json](https://github.com/FederatedAI/FATE/blob/master/python/fate_flow/examples/import_model.json).

For instance, the import configuration of guest(99) can be shown as below:

```
{
  "role": "guest",
  "party_id": 99,
  "model_id": "guest-99#host-100#model",
  "model_version": "fate_migration",
  "file": "/data/projects/fate/python/temp/guest#99#guest-99#host-100#model_fate_migration.zip"
}
```

Please fill in the role, the current party_id, the new model_id and model_version of the migration model, and the path of the compressed file of the migration model according to the actual situation. Please note that if you set two roles(host and arbiter eg.) in the same machine, you will need to execute the command twice with corresponding configuration. 

An example of submit command is: 

```bash
flow model import -c /data/projects/fate/python/fate_flow/examples/import_model.json
```

The following return is regarded as import success：

```json
{
  "retcode": 0,
  "retmsg": "success"
}
```

The migration task is now complete, and users can use the new model_id and model_version to submit the task to perform prediction tasks using the migrated model.