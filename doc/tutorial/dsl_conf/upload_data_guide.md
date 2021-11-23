# Upload Data Guide

[[中文](upload_data_guide.zh.md)]

Before start a modeling task, the data to be used should be uploaded.
Typically, a party is usually a cluster which include multiple nodes.
Thus, when we upload these data, the data will be allocated to those
nodes.

## Accepted Data Type

DataTransform(DataIO) module accepts the following input data format and transforms
them to desired output Table.

  - dense input format  
    input Table's value item is a list of single element, e.g. :

    ```
    1.0,2.0,3.0,4.5
    1.1,2.1,3.4,1.3
    2.4,6.3,1.5,9.0
    ```

  - svm-light input format  
    first item of input Table's value is label, following by a list of
    complex "feature\_id:value" items, e.g. :
    
    ``` 
    1 1:0.5 2:0.6
    0 1:0.7 3:0.8 5:0.2
    ```

  - tag input format  
    the input Table's value is a list of tag, data io module first
    aggregates all tags occurred in input table, then changes all input
    line to one-hot representation in sorting the occurred tags by
    lexicographic order, e.g. assume values is :
    
    ```    
    a c
    a b d
    ```
    
    after processing, the new values became: :

    ``` 
    1 0 1 0
    1 1 0 1
    ```

  - :tag:value input format: the input Table's value is a list of
    <tag:value>, like a mixed svm-light and tag input-format. data io
    module first aggregates all tags occurred in input table, then
    changes all input line to one-hot representation in sorting the
    occurred tags by lexicographic order, then fill the occur item with
    value. e.g. assume values is

    ```  
    a:0.2 c:1.5
    a:0.3 b:0.6 d:0.7
    ```
    
    after processing, the new values became: :

    ``` 
    0.2 0 0.5 0
    0.3 0.6 0 0.7
    ```

## Define upload data config file

Here is an example showing how to create a upload config file:

```json
{
  "file": "examples/data/breast_hetero_guest.csv",
  "table_name": "hetero_breast_guest",
  "namespace": "experiment",
  "head": 1,
  "partition": 8
}
```

Field Specifications:

1.  file: file path
2.  table\_name & namespace: Indicators for stored data table.
3.  head: Specify whether your data file include a header or not
4.  partition: Specify how many partitions used to store the data

## Upload Command

We use fate-flow to upload data. Starting at FATE ver1.5, [FATE-Flow
Client Command Line](https://github.com/FederatedAI/FATE-Flow/blob/main/doc/cli) is
recommended for interacting with FATE-Flow.

The command is as follows:

```bash
$ flow data upload -c examples/dsl/v2/upload/upload_conf.json
```

Meanwhile, user can still upload data using python script as in the
older
versions:

```bash
python ${your_install_path}fateflow/python/fate_flow/fate_flow_client.py -f upload -c examples/dsl/v2/upload/upload_conf.json
```

!!! Note
    
    This step is needed for every data-provide party(i.e. Guest and Host).

After running this command, the following information is shown if it is
success.

```json
{
    "data": {
        "board_url": "http://127.0.0.1:8080/index.html#/dashboard?job_id=202111111542373868350&role=local&party_id=0",
        "code": 0,
        "dsl_path": "/data/projects/fate/fateflow/jobs/202111111542373868350/job_dsl.json",
        "job_id": "202111111542373868350",
        "logs_directory": "/data/projects/fate/fateflow/logs/202111111542373868350",
        "message": "success",
        "model_info": {
            "model_id": "local-0#model",
            "model_version": "202111111542373868350"
        },
        "namespace": "experiment",
        "pipeline_dsl_path": "/data/projects/fate/fateflow/jobs/202111111542373868350/pipeline_dsl.json",
        "runtime_conf_on_party_path": "/data/projects/fate/fateflow/jobs/202111111542373868350/local/0/job_runtime_on_party_conf.json",
        "runtime_conf_path": "/data/projects/fate/fateflow/jobs/202111111542373868350/job_runtime_conf.json",
        "table_name": "breast_hetero_guest",
        "train_runtime_conf_path": "/data/projects/fate/fateflow/jobs/202111111542373868350/train_runtime_conf.json"
    },
    "jobId": "202111111542373868350",
    "retcode": 0,
    "retmsg": "success"
}
```

And as this output shown, table\_name and namespace have been listed,
which can be taken as input config in submit-runtime conf.
