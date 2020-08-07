# Fate Flow Client SDK Doc

[TOC]

## Usage

```python
from fate_flow.flowpy.client import FlowClient
client = Flowclient('10.1.2.3', 9000, 'v1')
```



## Job Operations

### Usage

```python
client.job.submit(conf_path, dsl_path)
```



### Functions

#### ```submit(conf_path, dsl_path)```

- *Description*：Submit a pipeline job.
- *Arguments*：

| No.  | Argument  |  Type  | Required |           Description           |
| :--: | :-------: | :----: | :------: | :-----------------------------: |
|  1   | conf_path | string |   Yes    | Runtime configuration file path |
|  2   | dsl_path  | string |   Yes    |          DSL file path          |



#### ```stop(job_id)```

- *Description*：Cancel or stop a specified job. 
- *Arguments*：

| No.  | Argument |  Type   | Required |  Description   |
| :--: | :------: | :-----: | :------: | :------------: |
|  1   |  job_id  | integer |   Yes    | A valid job id |



#### ```query(job_id=None, role=None, party_id=None, component_name=None, status=None)```

- *Description*：Query job information by filters.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |    No    | A valid job id |
|  2   |      role      | string  |    No    |      Role      |
|  3   |    party_id    | integer |    No    |    Party id    |
|  4   | component_name | string  |    No    | Component Name |
|  5   |     status     | string  |    No    |   Job Status   |



#### ```clean(job_id, role=None, party_id=None, component_name=None)```

- *Description*： Cancel all jobs in queue.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |    No    |      Role      |
|  3   |    party_id    | integer |    No    |    Party id    |
|  4   | component_name | string  |    No    | Component Name |



#### ```config(job_id, role, party_id, output_path)```

- *Description*：Download the configuration of a specified job.
- *Arguments*：

|  No  |  Argument   |  Type   | Required |      Description      |
| :--: | :---------: | :-----: | :------: | :-------------------: |
|  1   |   job_id    | integer |   Yes    |    A valid job id     |
|  2   |    role     | string  |   Yes    |         Role          |
|  3   |  party_id   | integer |   Yes    |       Party id        |
|  4   | output_path | string  |   Yes    | Specified Output Path |



#### ```log(job_id, output_path)```

- *Description*：Download log files of a specified job.
- *Arguments*：

| No.  |  Argument   |  Type   | Required |      Description      |
| :--: | :---------: | :-----: | :------: | :-------------------: |
|  1   |   job_id    | integer |   Yes    |    A valid job id     |
|  2   | output_path | string  |   Yes    | Specified Output Path |



#### ```list(limit=10)```

- *Description*：List jobs.
- *Arguments*：

| No.  | Argument |  Type   | Required |                Description                 |
| :--: | :------: | :-----: | :------: | :----------------------------------------: |
|  1   |  limit   | integer |    No    | Limit the number of results, default is 10 |



#### ```view(job_id=None, role=None, party_id=None, status=None)```

- *Description*：List jobs.

- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |    No    | A valid job id |
|  2   |      role      | string  |    No    |      Role      |
|  3   |    party_id    | integer |    No    |    Party id    |
|  4   | component_name | string  |    No    | Component Name |



#### `generate_dsl(train_dsl_path, version="1", cpn_file_path=None, cpn_list = None)`

- *Description*：A predict dsl generator.
- *Arguments*：

| No.  |    Argument    |     Type     | Required |                         Description                          |
| :--: | :------------: | :----------: | :------: | :----------------------------------------------------------: |
|  1   | train_dsl_path | string(path) |   Yes    |           User specifies the train dsl file path.            |
|  2   |    version     |    string    |    No    |         Specified version of dsl parser. Default 1.          |
|  3   | cpn_file_path  | string(path) |    No    | User specifies a file path which records the component list. |
|  4   |    cpn_list    |     list     |    No    |            User inputs a list of component names.            |





## Component Operations

### Usage

```python
client.component.parameters(job_id, role, party_id, component_name)
```



### Functions

#### ```parameters(job_id, role, party_id, component_name)```

- *Description*：Query the parameters of a specified component.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |   Yes    |      Role      |
|  3   |    party_id    | integer |   Yes    |    Party id    |
|  4   | component_name | string  |   Yes    | Component Name |



#### ```metric_all(job_id, role, party_id, component_name)```

- *Description*：Query all metric data.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |   Yes    |      Role      |
|  3   |    party_id    | integer |   Yes    |    Party id    |
|  4   | component_name | string  |   Yes    | Component Name |



#### ```metrics(job_id, role, party_id, component_name)```

- *Description*：Query all metric data.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |   Yes    |      Role      |
|  3   |    party_id    | integer |   Yes    |    Party id    |
|  4   | component_name | string  |   Yes    | Component Name |



#### ```metric_delete(date=None, job_id=None)```

- *Description*：Delete specified metric. 
- *Arguments*：

| No.  | Argument |  Type   | Required |                  Description                  |
| :--: | :------: | :-----: | :------: | :-------------------------------------------: |
|  1   |   date   | integer |   Yes    | An 8-Digit Valid Date, Format Like 'YYYYMMDD' |
|  2   |  job_id  | integer |   Yes    |                A valid job id                 |

`Notice`: If you input two optional arguments in the mean time, the 'date' argument will be detected in priority while the 'job_id' argument would be ignored.



#### ```output_model(job_id, role, party_id, component_name)```

- *Description*：Query a specified component model.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |   Yes    |      Role      |
|  3   |    party_id    | integer |   Yes    |    Party id    |
|  4   | component_name | string  |   Yes    | Component Name |



#### ```output_data(job_id, role, party_id, component_name, output_path, limit=10)```

- *Description*：Download the output data of a specified component.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |                Description                 |
| :--: | :------------: | :-----: | :------: | :----------------------------------------: |
|  1   |     job_id     | integer |   Yes    |               A valid job id               |
|  2   |      role      | string  |   Yes    |                    Role                    |
|  3   |    party_id    | integer |   Yes    |                  Party id                  |
|  4   | component_name | string  |   Yes    |               Component Name               |
|  5   |  output_path   | string  |   Yes    |      Specified Output directory path       |
|  6   |     limit      | integer |    No    | Limit the number of results, default is 10 |



#### ```output_data_table(job_id, role, party_id, component_name)```

- *Description*：View table name and namespace.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |     job_id     | integer |   Yes    | A valid job id |
|  2   |      role      | string  |   Yes    |      Role      |
|  3   |    party_id    | integer |   Yes    |    Party id    |
|  4   | component_name | string  |   Yes    | Component Name |



#### ```list(job_id)```

- *Description*： List components of a specified job.
- *Arguments*：

| No.  | Argument |  Type   | Required |  Description   |
| :--: | :------: | :-----: | :------: | :------------: |
|  1   |  job_id  | integer |   Yes    | A valid job id |





## Data Operations

### Usage

```python
client.data.download(conf_path)
```



### Functions

#### ```download(conf_path)```

- *Description*：Download Data Table.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```upload(conf_path, verbose=0, drop=0)```

- *Description*：Upload Data Table.
- *Arguments*：

| No.  | Argument  |  Type   | Required |                         Description                          |
| :--: | :-------: | :-----: | :------: | :----------------------------------------------------------: |
|  1   | conf_path | string  |   Yes    |                   Configuration file path                    |
|  2   |  verbose  | integer |    No    | Verbose mode, 0 (default) means 'disable', 1 means 'enable'  |
|  3   |   drop    | integer |    No    | If 'drop' is set to be 0 (defualt), when data had been uploaded before, current upload task would be rejected. If 'drop' is set to be 1, data of old version would be replaced by the latest version. |



#### ```upload_history(limit=10, job_id=None)```

- *Description*：Query Upload Table History.
- *Arguments*：

| No.  | Argument |  Type   | Required |                Description                 |
| :--: | :------: | :-----: | :------: | :----------------------------------------: |
|  1   |  limit   | integer |    No    | Limit the number of results, default is 10 |
|  2   |  job_id  | integer |    No    |               A valid job id               |





## Task Operations

### Usage

```python
client.task.list(limit=10)
```



### Functions

#### ```list(limit=10)```

- *Description*： List tasks.
- *Arguments*：

| No.  | Argument |  Type   | Required |                Description                 |
| :--: | :------: | :-----: | :------: | :----------------------------------------: |
|  1   |  limit   | integer |    No    | Limit the number of results, default is 10 |



#### ```query(job_id=None, role=None, party_id=None, component_name=None, status=None)```

- *Description*： Query task information by filters.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |   Description   |
| :--: | :------------: | :-----: | :------: | :-------------: |
|  1   |     job_id     | integer |    No    | A valid job id. |
|  2   |      role      | string  |    No    |      Role       |
|  3   |    party_id    | integer |    No    |    Party ID     |
|  4   | component_name | string  |    No    | Component Name  |
|  5   |     status     | string  |    No    |   Job Status    |





## Model Operations

### Usage

```python
client.model.load(conf_path)
```



### Functions

#### ```load(conf_path)```

- *Description*： Load model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```bind(conf_path)```

- *Description*： Bind model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```store(conf_path)```

- *Description*： Store model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```restore(conf_path)```

- *Description*： Restore model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```export(conf_path)```

- *Description*： Export model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |



#### ```imp(conf_path)```

- *Description*： Import model.
- *Arguments*：

| No.  | Argument  |  Type  | Required |       Description       |
| :--: | :-------: | :----: | :------: | :---------------------: |
|  1   | conf_path | string |   Yes    | Configuration file path |





## Table Operations

### Usage

```python
client.table.info(namespace, table_name)
```



### Functions

#### ```info(namespace, table_name)```

- *Description*： Query table information.
- *Arguments*：

| No.  |  Argument  |  Type  | Required | Description |
| :--: | :--------: | :----: | :------: | :---------: |
|  1   | namespace  | string |   Yes    |  Namespace  |
|  2   | table_name | string |   Yes    | Table Name  |



#### ```delete(namespace=None, table_name=None, job_id=None, role=None, party_id=None, component_name=None)```

- *Description*：Delete table.
- *Arguments*：

| No.  |    Argument    |  Type   | Required |  Description   |
| :--: | :------------: | :-----: | :------: | :------------: |
|  1   |   namespace    | string  |    No    |   Namespace    |
|  2   |   table_name   | string  |    No    |   Table Name   |
|  3   |     job_id     | integer |    No    | A valid job id |
|  4   |      role      | string  |    No    |      Role      |
|  5   |    party_id    | integer |    No    |    Party id    |
|  6   | component_name | string  |    No    | Component Name |





## Queue Operations

### Usage

```python
client.queue.clean()
```



### Functions

#### ```clean()```

- *Description*：Cancel all jobs in queue. 
- *Arguments*：None
