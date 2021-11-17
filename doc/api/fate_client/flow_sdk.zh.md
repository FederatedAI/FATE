# Fate Flow Client SDK 指南

[[ENG](../flow_sdk.zh.md)]

## 用法

``` sourceCode python
from flow_sdk.client import FlowClient
# use real ip address to initialize SDK
client = FlowClient('127.0.0.1', 9000, 'v1')
```

## Job 操作

### 用法

``` sourceCode python
client.job.submit(conf_path, dsl_path)
```

### 函数定义

#### `submit(conf_path, dsl_path)`

  - 介绍：提交执行pipeline任务。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |
| 2    | dsl_path  | string   | 是       | DSL文件路径      |

#### `stop(job_id)`

  - 介绍：取消或终止指定任务。
  - 参数：

| 编号 | 参数   | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------ | -------- | -------- | -------- |
| 1    | job_id | integer  | 是       | Job ID   |

#### `query(job_id=None, role=None, party_id=None, status=None)`

  - 介绍：检索任务信息。
  - 参数：

| 编号 | 参数     | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------- | -------- | -------- | -------- |
| 1    | job_id   | integer  | 否       | Job ID   |
| 2    | role     | string   | 否       | 角色     |
| 3    | party_id | integer  | 否       | Party id |
| 4    | status   | string   | 否       | 任务状态 |

#### `config(job_id, role, party_id, output_path)`

  - 介绍：下载指定任务的配置文件到指定目录。
  - 参数：

| 否   | 参数        | 参数类型 | 必要参数 | 参数介绍     |
| ---- | ----------- | -------- | -------- | ------------ |
| 1    | job_id      | integer  | 是       | Job ID       |
| 2    | role        | string   | 是       | 角色         |
| 3    | party_id    | integer  | 是       | Party id     |
| 4    | output_path | string   | 是       | 输出目录路径 |

#### `log(job_id, output_path)`

  - 介绍：下载指定任务的日志文件到指定目录。
  - 参数：

| 编号 | 参数        | 参数类型 | 必要参数 | 参数介绍     |
| ---- | ----------- | -------- | -------- | ------------ |
| 1    | job_id      | integer  | 是       | Job ID       |
| 2    | output_path | string   | 是       | 输出目录路径 |

#### `list(limit=10)`

  - 介绍：展示任务列表。
  - 参数：

| 编号 | 参数  | 参数类型 | 必要参数 | 参数介绍                 |
| ---- | ----- | -------- | -------- | ------------------------ |
| 1    | limit | integer  | 否       | 返回数量限制（默认：10） |

#### `view(job_id=None, role=None, party_id=None, status=None)`

  - 介绍：检索任务数据视图。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 否       | Job ID   |
| 2    | role=          | string   | 否       | 角色     |
| 3    | party_id       | integer  | 否       | Party id |
| 4    | component_name | string   | 否       | 组件名   |

#### `generate_dsl(train_dsl_path, cpn_file_path=None, cpn_list = None)`

  - 介绍：预测DSL文件生成器。
  - 参数：

| 编号 | 参数           | 参数类型     | 必要参数 | 参数介绍                         |
| ---- | -------------- | ------------ | -------- | -------------------------------- |
| 1    | train_dsl_path | string(path) | 是       | 用户指定组件名列表               |
| 2    | version        | string       | 否       | DSL解析器版本（默认：1）         |
| 3    | cpn_file_path  | string(path) | 否       | 用户指定带有组件名列表的文件路径 |
| 4    | cpn_list       | list         | 否       | 用户指定组件名列表               |

## Component 操作

### 用法

``` sourceCode python
client.component.parameters(job_id, role, party_id, component_name)
```

### 函数定义

#### `parameters(job_id, role, party_id, component_name)`

  - 介绍：检索指定组件的参数。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | 角色     |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

#### `metric_all(job_id, role, party_id, component_name)`

  - 介绍：检索指定任务的所有metric数据。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | 角色     |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

#### `metrics(job_id, role, party_id, component_name)`

  - 介绍：检索指定任务指定组件的metric数据。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | 角色     |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

#### `metric_delete(date=None, job_id=None)`

  - 介绍：删除指定metric数据。
  - 参数：

| 编号 | 参数   | 参数类型 | 必要参数 | 参数介绍                 |
| ---- | ------ | -------- | -------- | ------------------------ |
| 1    | date   | integer  | 是       | 8位日期, 形如 'YYYYMMDD' |
| 2    | job_id | integer  | 是       | Job ID                   |

`Notice`： If you input two optional 参数s in the mean time, the 'date' 参数
will be detected in priority while the 'job_id' 参数 would be ignored.

#### `output_model(job_id, role, party_id, component_name)`

  - 介绍：检索指定组件模型。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | role=    |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

#### `output_data(job_id, role, party_id, component_name, output_path, limit=10)`

  - 介绍：下载指定组件的输出数据。
  - 参数：

| 编号 | 参数              | 参数类型    | 必要参数 | 参数介绍                    |
| -- | --------------- | ------- | ---- | ----------------------- |
| 1  | job_id         | integer | 是    | Job ID                  |
| 2  | role            | string  | 是    | 角色                      |
| 3  | party_id       | integer | 是    | Party id                |
| 4  | component_name | string  | 是    | 组件名                     |
| 5  | output_path    | string  | 是    | 输出目录路径                  |
| 6  | limit           | integer | 否    | 返回结果数量限制（默认：-1，指返回所有数据） |

#### `output_data_table(job_id, role, party_id, component_name)`

  - 介绍：查看数据表名及命名空间。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | 角色     |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

#### `list(job_id)`

  - 介绍： 展示指定任务的组件列表。
  - 参数：

| 编号 | 参数   | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------ | -------- | -------- | -------- |
| 1    | job_id | integer  | 是       | Job ID   |

#### `get_summary(job_id, role, party_id, component_name)`

  - 介绍：获取指定组件的概要。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 是       | Job ID   |
| 2    | role           | string   | 是       | 角色     |
| 3    | party_id       | integer  | 是       | Party id |
| 4    | component_name | string   | 是       | 组件名   |

## Data 操作

### 用法

``` sourceCode python
client.data.download(conf_path)
```

### 函数定义

#### `download(conf_path)`

  - 介绍：下载数据表。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |

#### `upload(conf_path, verbose=0, drop=0)`

  - 介绍：上传数据表。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍                                                     |
| ---- | --------- | -------- | -------- | ------------------------------------------------------------ |
| 1    | conf_path | string   | 是       | 任务配置文件路径                                             |
| 2    | verbose   | integer  | 否       | 如果赋值为1，用户将在控制台获得上传进度（默认为0）           |
| 3    | drop      | integer  | 否       | 如果赋值为1，旧版已上传数据将被新上传的同名数据替换（默认为0） |

#### `upload_history(limit=10, job_id=None)`

  - 介绍：检索上传数据历史。
  - 参数：

| 编号 | 参数   | 参数类型 | 必要参数 | 参数介绍                     |
| ---- | ------ | -------- | -------- | ---------------------------- |
| 1    | limit  | integer  | 否       | 返回结果数量限制（默认：10） |
| 2    | job_id | integer  | 否       | Job ID                       |

## Task 操作

### 用法

``` sourceCode python
client.task.list(limit=10)
```

### 函数定义

#### `list(limit=10)`

  - 介绍： 展示Task列表。
  - 参数：

| 编号 | 参数    | 参数类型    | 必要参数 | 参数介绍            |
| -- | ----- | ------- | ---- | --------------- |
| 1  | limit | integer | 否    | 返回结果数量限制（默认：10） |

#### `query(job_id=None, role=None, party_id=None, component_name=None, status=None)`

  - 介绍： 检索Task信息。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | job_id         | integer  | 否       | Job ID.  |
| 2    | role           | string   | 否       | 角色     |
| 3    | party_id       | integer  | 否       | Party ID |
| 4    | component_name | string   | 否       | 组件名   |
| 5    | status         | string   | 否       | 任务状态 |

## Model 操作

### 用法

``` sourceCode python
client.model.load(conf_path)
```

### 函数定义

#### `load(conf_path=None, job_id=None)`

  - 介绍：加载模型。如果 `dsl_version == 2` 则需要先 `deploy` 模型。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 否       | 任务配置文件路径 |
| 2    | job_id    | string   | 否       | Job ID           |

#### `bind(conf_path, job_id=None)`

  - 介绍： 绑定模型。如果 `dsl_version == 2` 则需要先 `deploy` 模型。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |
| 2    | job_id    | string   | 否       | Job ID           |

#### `export_model(conf_path, to_database=False)`

  - 介绍：
导出模型。
  - 参数：

| 编号 | 参数        | 参数类型 | 必要参数 | 参数介绍                                                     |
| ---- | ----------- | -------- | -------- | ------------------------------------------------------------ |
| 1    | conf_path   | string   | 是       | 任务配置文件路径                                             |
| 2    | to_database | bool     | 否       | 如果指定且有可用的数据库环境，fate flow将从根据任务配置文件将模型导出到数据库中。 |

#### `import_model(conf_path, from_database=False)`

  - 介绍：
导入模型。
  - 参数：

| 编号 | 参数          | 参数类型 | 必要参数 | 参数介绍                                                     |
| ---- | ------------- | -------- | -------- | ------------------------------------------------------------ |
| 1    | conf_path     | string   | 是       | 任务配置文件路径                                             |
| 2    | from_database | bool     | 否       | 如果指定且有可用的数据库环境，fate flow将从根据任务配置文件从数据库中导入模型。 |

#### `migrate(conf_path, to_database=False)`

  - 介绍： 迁移模型。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |

#### `tag_list(job_id)`

  - 介绍： 展示模型的标签列表。
  - 参数：

| 编号 | 参数   | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------ | -------- | -------- | -------- |
| 1    | job_id | integer  | 是       | Job ID   |

#### `tag_model(job_id, tag_name, remove=False)`

  - 介绍： 对模型添加标签。
  - 参数：

| 编号 | 参数     | 参数类型 | 必要参数 | 参数介绍                                               |
| ---- | -------- | -------- | -------- | ------------------------------------------------------ |
| 1    | job_id   | integer  | 是       | Job ID                                                 |
| 2    | tag_name | string   | 是       | 标签名                                                 |
| 3    | remove   | bool     | 否       | 如果指定，带有指定标签名的标签将被模型的标签列表中移除 |

#### `deploy(model_id, model_version=None, cpn_list=None, predict_dsl=None, components_checkpoint=None)`

  - 介绍： 配置模型预测dsl。
  - 参数：

| 编号 | 参数                  | 参数类型 | 必要参数 | 参数介绍        |
| ---- | --------------------- | -------- | -------- | --------------- |
| 1    | model_id              | string   | 是       | 模型ID          |
| 2    | model_version         | string   | 是       | 模型版本        |
| 3    | cpn_list              | list     | 否       | 组件列表        |
| 4    | predict_dsl           | dict     | 否       | 预测DSL         |
| 5    | components_checkpoint | dict     | 否       | 指定 checkpoint |

#### `get_predict_dsl(model_id, model_version)`

  - 介绍： 获取模型预测dsl。
  - 参数：

| 编号 | 参数          | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------------- | -------- | -------- | -------- |
| 1    | model_id      | string   | 是       | 模型ID   |
| 2    | model_version | string   | 是       | 模型版本 |

#### `get_predict_conf(model_id, model_version)`

  - 介绍： 获取模型预测conf模板。
  - 参数：

| 编号 | 参数          | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------------- | -------- | -------- | -------- |
| 1    | model_id      | string   | 是       | 模型ID   |
| 2    | model_version | string   | 是       | 模型版本 |

#### `get_model_info(model_id=None, model_version=None, role=None, party_id=None, query_filters=None, **kwargs)`

  - 介绍： 获取模型信息。
  - 参数：

| 编号 | 参数          | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ------------- | -------- | -------- | -------- |
| 1    | model_id      | string   | 否       | 模型ID   |
| 2    | model_version | string   | 是       | 模型版本 |
| 3    | role          | string   | 否       | 角色名   |
| 4    | party_id      | string   | 否       | Party ID |
| 5    | query_filters | list     | 否       | 检索字段 |

#### `homo_convert(conf_path)`

  - 介绍： 基于横向训练的模型，生成其他ML框架的模型文件。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |

#### `homo_deploy(conf_path)`

  - 介绍： 将横向训练之后使用homo_convert功能生成的模型部署到在线推理系统中，当前支持创建基于KFServing的推理服务。
  - 参数：

| 编号 | 参数      | 参数类型 | 必要参数 | 参数介绍         |
| ---- | --------- | -------- | -------- | ---------------- |
| 1    | conf_path | string   | 是       | 任务配置文件路径 |

## Tag 操作

### 用法

``` sourceCode python
client.tag.create(tag_name, desc)
```

### 函数定义

#### `create(tag_name, tag_desc=None)`

  - 介绍：创建标签。
  - 参数：

| 编号 | 参数     | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------- | -------- | -------- | -------- |
| 1    | tag_name | string   | 是       | 标签名   |
| 2    | tag_desc | string   | 否       | 标签介绍 |

#### `update(tag_name, new_tag_name=None, new_tag_desc=None)`

  - 介绍： 更新标签信息。
  - 参数：

| 编号 | 参数         | 参数类型 | 必要参数 | 参数介绍   |
| ---- | ------------ | -------- | -------- | ---------- |
| 1    | tag_name     | string   | 是       | 标签名     |
| 2    | new_tag_name | string   | 否       | 新标签名   |
| 3    | new_tag_desc | string   | 否       | 新标签介绍 |

#### `list(limit=10)`

  - 介绍： 展示标签列表。
  - 参数：

| 编号 | 参数  | 参数类型 | 必要参数 | 参数介绍                     |
| ---- | ----- | -------- | -------- | ---------------------------- |
| 1    | limit | integer  | 否       | 返回结果数量限制（默认：10） |

#### `query(tag_name, with_model=False)`

  - 介绍： 检索标签。
  - 参数：

| 编号 | 参数       | 参数类型 | 必要参数 | 参数介绍                               |
| ---- | ---------- | -------- | -------- | -------------------------------------- |
| 1    | tag_name   | string   | 是       | 标签名                                 |
| 2    | with_model | bool     | 否       | 如果指定，具有该标签的模型信息将被展示 |

#### `delete(tag_name)`

  - 介绍： 删除标签。
  - 参数：

| 编号 | 参数     | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------- | -------- | -------- | -------- |
| 1    | tag_name | string   | 是       | 标签名   |

## Table 操作

### 用法

``` sourceCode python
client.table.info(namespace, table_name)
```

### 函数定义

#### `info(namespace, table_name)`

  - 介绍： 检索数据表信息。
  - 参数：

| 编号 | 参数       | 参数类型 | 必要参数 | 参数介绍 |
| ---- | ---------- | -------- | -------- | -------- |
| 1    | namespace  | string   | 是       | 命名空间 |
| 2    | table_name | string   | 是       | 数据表名 |

#### `delete(namespace=None, table_name=None, job_id=None, role=None, party_id=None, component_name=None)`

  - 介绍：删除指定数据表。
  - 参数：

| 编号 | 参数           | 参数类型 | 必要参数 | 参数介绍 |
| ---- | -------------- | -------- | -------- | -------- |
| 1    | namespace      | string   | 否       | 命名空间 |
| 2    | table_name     | string   | 否       | 数据表名 |
| 3    | job_id         | integer  | 否       | Job ID   |
| 4    | role=          | string   | 否       | 角色     |
| 5    | party_id       | integer  | 否       | Party id |
| 6    | component_name | string   | 否       | 组件名   |

## Queue 操作

### 用法

``` sourceCode python
client.queue.clean()
```

### 函数定义

#### `clean()`

  - 介绍：取消所有在队列中的Job。
  - 参数：无
