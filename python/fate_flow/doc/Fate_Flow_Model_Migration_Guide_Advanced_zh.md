# 模型迁移功能使用指南

[TOC]

模型迁移功能使得模型文件复制拷贝到不同party id的集群依然可用，以下两种场景需要做模型迁移：
1. 模型生成参与方任何一方的集群, 重新部署且部署后集群的party id变更, 例如源arbiter-10000#guest-9999#host-10000, 改为arbiter-10000#guest-99#host-10000
2. 将模型文件从源集群复制到目标集群，需要在目标集群使用

迁移流程如下：

## 转移模型文件

请将源集群fate flow服务所在机器生成的模型文件（包括以model id为命名的目录）进行打包并转移到目标集群fate flow所在机器中，请将模型文件转移至固定目录中：

```shell
$FATE_PATH/model_local_cache
```

文件夹转移即可，如果是通过压缩打包进行的转移，请在转移后将模型文件解压到模型所在目录中。

## 迁移前的准备工作

### 说明
1. 安装支持模型迁移的客户端fate-client，只有fate 1.5.1及其以上版本支持
2. 如果希望在一方发起迁移任务，自动分发任务到所有参与方执行，那么需要在源集群和目标集群都完成迁移前的准备工作，并且只能在源集群执行迁移任务
3. 若不需要同时一步完成所有参与方执行，而是分别在每个参与方单独执行迁移任务，那么可以只在目标集群完成迁移前的准备工作，然后在目标集群执行迁移任务

### 1. 安装fate-client

请在装有1.5.1及其以上版本fate的机器中进行安装：

安装命令：

```shell
# 进入FATE的安装路径，例如/data/projects/fate
cd $FATE_PATH/
# 进入FATE PYTHON的虚拟环境
source bin/init_env.sh
# 执行安装
pip install ./python/fate_client
```

安装完成之后，在命令行键入`flow` 并回车，获得如下返回即视为安装成功：

```shell
Usage: flow [OPTIONS] COMMAND [ARGS]...

  Fate Flow Client

Options:
  -h, --help  Show this message and exit.

Commands:
  component  Component Operations
  data       Data Operations
  init       Flow CLI Init Command
  job        Job Operations
  model      Model Operations
  queue      Queue Operations
  table      Table Operations
  tag        Tag Operations
  task       Task Operations
```



### 2. fate-client初始化：

在使用fate-client之前需要对其进行初始化，推荐使用fate的配置文件进行初始化，初始化命令如下：

```shell
# 进入FATE的安装路径，例如/data/projects/fate
cd $FATE_PATH/
# 指定fate服务配置文件进行初始化
flow init -c ./conf/service_conf.yaml
```

获得如下返回视为初始化成功：

```json
{
    "retcode": 0,
    "retmsg": "Fate Flow CLI has been initialized successfully."
}
```

如果fate-client的安装机器与FATE-Flow不在同一台机器上，请使用IP地址和端口号进行初始化，初始化命令如下：

```shell
# 进入FATE的安装路径，例如/data/projects/fate
cd $FATE_PATH/
# 指定fate的IP地址和端口进行初始化
flow init --ip 192.168.0.1 --port 9380
```


## 执行迁移任务

### 说明
1. 执行迁移任务是将源模型文件根据迁移任务配置文件修改model_id、model_version以及模型内涉及`role`和`party_id`的内容进行替换
2. 提交并执行任务的集群必须完成上述迁移准备
3. 如果希望在一方发起迁移任务，自动分发任务到所有参与方执行，只能在源集群执行迁移任务
3. 若不需要同时一步完成所有参与方执行，而是分别在每个参与方单独执行迁移任务，可以在源集群

### 1. 修改配置文件

在源集群（机器）中根据实际情况对迁移任务的配置文件进行修改，如下为迁移任务示例配置文件 [migrate_model.json](https://github.com/FederatedAI/FATE/blob/master/python/fate_flow/examples/migrate_model.json)

```json
{
  "job_parameters": {
    "federated_mode": "MULTIPLE"
  },
  "role": {
    "guest": [9999],
    "arbiter": [10000],
    "host": [10000]
  },
  "migrate_initiator": {
    "role": "guest",
    "party_id": 99
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

请将上述配置内容保存到服务器中的某一位置进行修改。

以下为对该配置中的参数的解释说明：

1. **`job_parameters`**：该参数中的`federated_mode`有两个可选参数，分别为`MULTIPLE` 及`SINGLE`。如果设置为`MULTIPLE`，则将任务分发到`execute_party`中指定的参与方执行任务；如果设置为`SINGLE`，则该迁移任务只会在提交迁移任务的本方执行。
2. **`role`**：该参数填写生成原始模型的参与方`role`及其对应的`party_id`信息。
3. **`migrate_initiator`**：该参数用于指定迁移后的模型的任务发起方信息，分别需指定发起方的`role`与`party_id`。
4. **`migrate_role`**：该参数用于指定迁移后的模型的参与方`role`及`party_id`信息。
5. **`execute_party`**：该参数用于指定需要执行迁移的`role`及`party_id`信息, 如该`party_id`为源集群`party_id`。
6. **`model_id`**：该参数用于指定需要被迁移的原始模型的`model_id`。
7. **`model_version`**：该参数用于指定需要被迁移的原始模型的`model_version`。
8. **`unify_model_version`**：此参数为非必填参数，该参数用于指定新模型的`model_version`。若未提供该参数，新模型将以迁移任务的`job_id`作为其新`model_version`。

上述配置文件举例说明：
1. 源模型的参与方为guest: 9999, host: 10000, arbiter: 10000, 将模型迁移成参与方为guest: 99, host: 100, arbiter: 100, 且新发起方为guest: 99
2. `federated_mode`: `MULTIPLE`表示在9999一方发起, 10000也会同时执行迁移动作;
3. 若考虑到风险，可以设置`federated_mode`: `SINGLE`



## 2. 提交迁移任务

如果需进行联邦迁移（即多方同时迁移），需将**`job_parameters`**设置为`MULTIPLE`并在`guest`方提交任务；如果只需要在本方（arbiter、guest、host均可）中进行迁移，可将**`job_parameters`**设置为`SINGLE`，并提交任务。

迁移任务需使用FATE Flow CLI v2进行提交，示例执行命令如下：

```bash
flow model migrate -c /data/projects/fate/python/fate_flow/examples/migrate_model.json
```



## 3. 任务执行结果

如下为实际迁移任务的配置文件内容：

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

该任务实现的是，将party_id为9999（guest），10000（host）的集群中的model_id为guest-9999#host-10000#model，model_version为202010291539339602784的模型迁移到party_id为99（guest），100（host）的集群中，且用户自定义新模型的model_version为fate_migration。模式设置为联邦迁移意味着需要在guest方（9999）中提交任务，任务提交后，execute_party中的guest方（9999）及host方（10000）将同时执行迁移任务。



如下为迁移成功的后得到的返回结果：

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

任务成功执行后，执行方的机器中都会生成一份迁移后模型压缩文件，该文件路径可以在返回结果中得到。如上，guest方（9999）的迁移后模型文件路径为：/data/projects/fate/temp/fate_flow/guest#99#guest-99#host-100#model_fate_migration.zip，host方（10000）的迁移后模型文件路径为：/data/projects/fate/temp/fate_flow/host#100#guest-99#host-100#model_fate_migration.zip。新的model_id与model_version同样可以从返回中获得。



## 4. 转移文件并导入

迁移任务成功之后，请手动将新生成的模型压缩文件转移到对应的迁往机器上。例如：第三点中guest方（9999）生成的新模型压缩文件需要被转移到guest（99）机器上。压缩文件可以放在对应机器上的任意位置，接下来需要配置模型的导入任务，配置文件请见[import_model.json](https://github.com/FederatedAI/FATE/blob/master/python/fate_flow/examples/import_model.json)。

下面举例介绍在guest（99）中导入迁移后模型的配置文件：

```
{
  "role": "guest",
  "party_id": 99,
  "model_id": "guest-99#host-100#model",
  "model_version": "fate_migration",
  "file": "/data/projects/fate/python/temp/guest#99#guest-99#host-100#model_fate_migration.zip"
}
```

请根据实际情况对应填写角色role，当前本方party_id，迁移模型的新model_id及model_version，以及迁移模型的压缩文件所在路径。

如下为使用FATE Flow CLI v2提交导入模型的示例命令：

```bash
flow model import -c /data/projects/fate/python/fate_flow/examples/import_model.json
```

得到如下返回视为导入成功：

```json
{
  "retcode": 0,
  "retmsg": "success"
}
```

迁移任务至此完成，用户可使用新的model_id及model_version进行任务提交，以利用迁移后的模型执行预测任务。