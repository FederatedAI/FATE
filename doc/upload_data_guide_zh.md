## 上传数据指南
在开始建模任务之前，应上传要使用的数据。通常来说，一个参与方是包含多个节点的集群。因此，当我们上传数据时，这些数据将被分配给这些节点。

### 接受的数据类型

DataIO模块接受以下输入数据格式，并将其转换为所需的输出DTable。
* 稠密输入格式：输入的Dtable值是一个包含单个元素的列表。(例如："1.0, 2.0, 3.0, 4.5")
* svm-light输入格式：输入的Dtable值的第一项是label，其后是一个由键值对"feature-id:value"组成的列表。(例如："1 1:0.5 2:0.6")
* tag输入格式：输入的Dtable值是一个由tag组成的列表，DataIO模块首先统计所有在输入表中出现过的tag，然后将这些tag按字典序排序，并将它们转换成one-hot表示。(例如：假设输入是"a c"，"a b d"，经过处理，得到输出"1 0 1 0", "1 1 0 1")
* tag:value输入格式：输入的Dtable值是一个由键值对"tag:value"组成的列表，类似于svm-light输入格式和tag输入格式的结合。DataIO模块首先统计所有在输入表中出现过的tag，然后将这些tag按字典序排序。排序后的结果作为输出数据的列名，某条数据的每个tag对应的value则作为该条数据在相应列上的值。若该条数据的某个tag没有值，则填入0补充。(例如，假设输入是"a:0.2 c:1.5", "a:0.3 b:0.6 d:0.7"，经过处理，得到输出"0.2 0 1.5 0", "0.3 0.6 0 0.7")

### 定义上传数据配置文件

下面是一个说明如何创建上传配置文件的示例：
```json
{
  "file": "examples/data/breast_b.csv",
  "head": 1,
  "partition": 10,
  "work_mode": 0,
  "table_name": "hetero_breast_b",
  "namespace": "hetero_guest_breast"
}
```

字段说明：
1. file: 文件路径
2. head: 指定数据文件是否包含表头
3. partition: 指定用于存储数据的分区数
4. work_mode: 指定工作模式，0代表单机版，1代表集群版
5. table_name&namespace: 存储数据表的标识符号

### 上传命令
使用fate-flow上传数据。命令如下：
> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_test/upload_data.json

注：每个提供数据的集群（即guest和host）都需执行此步骤
运行此命令后，如果成功，将显示以下信息：

```json
{
    "data": {
        "namespace": "breast_hetero",
        "pid": 74684,
        "table_name": "breast_b"
    },
    "jobId": "20190801152750392991_436",
    "meta": null,
    "retcode": 0,
    "retmsg": "success",
    "created_at": "2019-08-01 15:27:50"
}
```
如输出所示，table_name和namespace已经列出，可以在submit-runtime.conf配置文件中作为输入配置。
