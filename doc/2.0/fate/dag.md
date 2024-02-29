# DAG Usage Guide
## 1. Introduction
This document primarily explains how to define a federated learning modeling task using DAG (Directed Acyclic Graph) since FATE-v2.0.

## 2. DAG Definition
This section goes through field definitions of a DAG file. In FATE-v2.0, DAG should be submitted as YAML.

Example DAG

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

### 2.1 Top-level Fields
#### 2.1.1 dag
Description file of the entire DAG.

#### 2.1.2 schema_version
Reserved field representing corresponding version number.

#### 2.1.3 kind
Protocol type. The DAG of fate-v2.0 is a universal workflow description file for federated learning tasks, supporting various protocols. The default is fate-v2.0 protocol, namely "fate". To represent other protocols using fate-v2.0 DAG, modify this field accordingly.

### 2.2 Sub-fields under DAG
Sub-fields consist of five components: parties, conf, stage, tasks, and party_tasks.

#### 2.2.1 parties
`parties` represents participants and is a list where each element is a party description object. Each party object consists of role and party_id.

```yaml
- party_id: ['9999']
  role: guest
```

* role
`role`represents the role in federated modeling, common roles include "guest", "host", "arbiter".

* party_id
A list representing which party_ids take this role.

#### 2.2.3 conf
Job-level task parameters. For details, refer to the [documentation](https://github.com/FederatedAI/FATE-Flow/blob/main/doc/job_scheduling.md).

#### 2.2.4 stage
Represents the stage of job execution, possible values include "default", "train", "predict", "cross_validation".
- default: Suitable for components that do not need to distinguish between execution stages, such as psi, union, evaluation, etc.
- train: Suitable for feature engineering, training algorithms, etc., indicating that the component runs the training process.
- predict: Suitable for feature engineering, training algorithms, etc., indicating that the component runs the prediction process.
- cross_validation: Used for training algorithms, indicating cross-validation is needed.

Note that `stage` is a job-level status. When a specific task execution stage is not specified, it inherits the job-level status. 
Otherwise, task-level specification takes precedence.

#### 2.2.5 tasks
General task configuration, as dictionary, where the key is the task name and the value is its general description, consisting of `component_ref`, `dependent_tasks`, `parameters`, `inputs`, `outputs`, `parties`, `conf`, and `stage`.

###### 2.2.5.1 component_ref
The component reference name that current task will call, given in string. Details on components can be found in the [component description document](./components/README.md).

###### 2.2.5.2 dependent_tasks
Upstream tasks that current depends on, given in list.

```yaml
dependent_tasks: [psi_0]
```

###### 2.2.5.3 parameters
General parameter configuration for current task.

```yaml
parameters:
  max_depth: 3
  num_trees: 2
  ...
```

###### 2.2.5.4 inputs
Inputs for current task, including upstream task outputs as inputs, direct inputs from the model warehouse, and direct inputs from the data warehouse (backup field).
- inputs: in dictionary type, where the key is data or model, and the value is the port connection relationship.
- Port connection relationship: in dictionary type, where the key is the input key corresponding to the component of current task, such as train_data, validate_data, input_data, etc., and the value is an input description object.
- Input description: upstream task outputs as inputs, direct inputs from the model warehouse, direct inputs from the data warehouse.


1. [x] Upstream task outputs as inputs:

    Using `task_output_artifact` to represent upstream task outputs as inputs.
    - producer_tasks: name of upstream tasks
    - output_artifact_key: specifies which output port of the upstream task serves as input
    - parties: optional, indicating which parties should use this input. If not specified, all parties' inputs for the task are consistent. Otherwise, specification takes effect. User may feed asymmetrical inputs to different parties through this key.

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

2. [x] Direct input from the model warehouse:

    Using model_warehouse to represent direct input from the model warehouse, mainly used for model inputs during the prediction phase.
    - model_id: model ID
    - model_version: model version
    - producer_task: name of the training task corresponding to (model_id, model_version)
    - output_artifact_key: model output name corresponding to (model_id, model_version).
    - parties: optional, indicating which parties should use this input. If not specified, all parties' inputs for the task are consistent. Otherwise, specification takes effect. User may feed asymmetrical inputs to different parties through this key.

###### 2.2.5.5 outputs
Optional, representing the component's outputs. Not required when scheduling with the fate-v2.0 protocol, currently only used for configuration in the interconnection protocol.

The output is currently set to dictionary type, where keys include data, model, and metric outputs, and value format includes three fields: output_artifact_key_alias, output_artifact_key_alias, parties.

- parties
`partied` involved in this task, in format consistent with job-level parties configuration. If specified by user, the participating parties during task execution are set accordingly. Otherwise, job-level setting will be inherited.

- conf
Configuration at task level, given in dictionary.

- stage
`stage` of task execution. If specified, given value takes precedence; otherwise, job-level setting is inherited.

#### 2.2.6 party_tasks
party_tasks represent personalized configurations for each party, given in dictionary, where the key is an alias indicating which sites will use this customized setting, and the value includes parties, tasks, conf.

```yaml
guest_9999:
  parties:
  - party_id: ['9999']
    role: guest
  tasks:
    reader_0:
      parameters: {name: breast_hetero_guest, namespace: experiment}
```
- parties
Indicates which parties are involved.

- conf
- Dictionary type, customized task configuration during task execution for each party.

- tasks
- Dictionary type, where the key represents which tasks will be executed, and the value represents specific task configuration, consisting of conf and parameters.
Conf is a dictionary, representing configuration for corresponding parties when running current task, and parameters are algorithm parameters for the task.

## 3. Prediction Task DAG
This section introduces DAG for pure prediction tasks. Compared to training tasks, there are fewer modifications needed for the prediction DAG. Below shows how to modify few places in training DAG to create a prediction DAG.

Prediction DAG Example
```yaml
dag:
  conf:
    model_warehouse: {model_id: ${train_task model_id}, model_version: ${train_task model_version}}
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
        reader_1:
          parameters: {name: breast_hetero_guest, namespace: experiment}
    host_10000:
      parties:
      - party_id: ['10000']
        role: host
      tasks:
        reader_1:
          parameters: {name: breast_hetero_host, namespace: experiment}
  stage: predict
  tasks:
    psi_0:
      component_ref: psi
      dependent_tasks: [reader_1]
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
              producer_task: reader_1
      parameters: {}
      stage: default
    reader_1:
      component_ref: reader
      parameters: {}
      stage: default
    sbt_0:
      component_ref: hetero_secureboost
      dependent_tasks: [psi_0]
      inputs:
        data:
          test_data:
            task_output_artifact:
            - output_artifact_key: output_data
              parties:
              - party_id: ['9999']
                role: guest
              - party_id: ['10000']
                role: host
              producer_task: psi_0
        model:
          input_model:
            model_warehouse:
              output_artifact_key: output_model
              parties:
              - party_id: ['9999']
                role: guest
              - party_id: ['10000']
                role: host
              producer_task: sbt_0
      parameters:
        max_depth: 3
        num_trees: 2
        ...
schema_version: 2.1.0
```

- Step1: Change the job-level stage in DAG to "predict"
- Step2: Remove unused components from `tasks`, as well as from `party_tasks`. Note that removing components may cause changes in component dependencies and inputs of downstream components, which need to be modified accordingly. As shown by modifications made on `eval_0` and `reader_0` components in the example.
- Step3: Add needed components to `tasks` and `party_tasks`, and modify fields and inputs of dependent downstream components accordingly.

```yaml
tasks:
  psi_0:
    component_ref: psi
    dependent_tasks: [reader_1]
  reader_1:
    component_ref: reader
    parameters: {}
    stage: default
```
- Step4: For prediction tasks, if models generated during the training phase are to be used, configure `model_warehouse` field in the job conf. Fill in `model_id` and `model_version` from the training task, and proceed to Step5.

```yaml
conf:
  model_warehouse: {model_id: ${train_task model_id}, model_version: ${train task model_version}}
```

- Step5: In tasks where trained models needed, add model inputs. Set `producer_task` to the name of training task component, and `output_artifact_key` to the corresponding model output field of the, and fill in the parties field as needed (because some third-party components may only have model inputs from guest/host during the prediction phase)
```yaml
inputs:
  model:
    input_model:
      model_warehouse:
        output_artifact_key: output_model
        parties:
        - party_id: ['9999']
          role: guest
        - party_id: ['10000']
          role: host
        producer_task: sbt_0
```

Once modifications above done, this DAG configuration may be used for running prediction task.

