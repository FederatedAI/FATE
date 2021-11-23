# 上传数据指南

[[ENG](upload_data_guide.md)]

在开始建模任务之前，应上传要使用的数据。通常来说，一个参与方是包含多个节点的集群。因此，当我们上传数据时，这些数据将被分配给这些节点。

## 接受的数据类型

DataTransform(DataIO)模块接受以下输入数据格式，并将其转换为所需的输出Table。

  - 稠密输入格式  
    输入的Table值是一个包含单个元素的列表，例如:

    ```
    1.0,2.0,3.0,4.5
    1.1,2.1,3.4,1.3
    2.4,6.3,1.5,9.0
    ```

  - svm-light输入格式  
    输入的Table值的第一项是label，其后是一个由键值对"feature-id:value"组成的列表，例如:

    ``` 
    1 1:0.5 2:0.6
    0 1:0.7 3:0.8 5:0.2
    ```

  - tag
    输入格式  
    输入的Table值是一个由tag组成的列表，DataIO模块首先统计所有在输入表中出现过的tag，然后将这些tag按字典序排序，并将它们转换成one-hot表示。例如: 
    
    假设值是:

    ``` 
    a c
    a b d
    ```
    
    经过处理, 新的值为:

    ``` 
    1 0 1 0
    1 1 0 1
    ```


  - :tag:value 输入格式:
    输入的Table值是一个由键值对"tag:value"组成的列表，类似于svm-light输入格式和tag输入格式的结合。DataIO模块首先统计所有在输入表中出现过的tag，然后将这些tag按字典序排序。排序后的结果作为输出数据的列名，某条数据的每个tag对应的value则作为该条数据在相应列上的值。若该条数据的某个tag没有值，则填入0补充。例如,
    
    假设值是:

    ```
    a:0.2 c:1.5
    a:0.3 b:0.6 d:0.7
    ```
    
    经过处理, 新的值为:

    ``` 
    0.2 0 0.5 0
    0.3 0.6 0 0.7
    ```

## 定义上传数据配置文件

下面是一个说明如何创建上传配置文件的示例：

```json
{
  "file": "examples/data/breast_hetero_guest.csv",
  "table_name": "hetero_breast_guest",
  "namespace": "experiment",
  "head": 1,
  "partition": 8
}
```

字段说明：

1.  file: 文件路径
2.  table\_name\&namespace: 存储数据表的标识符号
3.  head: 指定数据文件是否包含表头
4.  partition: 指定用于存储数据的分区数

## 上传命令

使用fate-flow上传数据。从FATE-1.5开始，推荐使用 [FATE-Flow Client Command
Line](https://github.com/FederatedAI/FATE-Flow/blob/main/doc/cli)
执行FATE-Flow任务。

上传数据命令如下：

```bash
$ flow data upload -c examples/dsl/v2/upload/upload_conf.json
```

同时，用户也可使用旧版的python脚本方式上传数据：

```bash
$ python ${your_install_path}/fateflow/python/fate_flow/fate_flow_client.py -f upload -c examples/dsl/v2/upload/upload_conf.json
```

!!! Note

    每个提供数据的集群（即guest和host）都需执行此步骤

运行此命令后，如果成功，将显示以下信息：

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

如输出所示，table\_name和namespace已经列出，可以在submit-runtime.conf配置文件中作为输入配置。
