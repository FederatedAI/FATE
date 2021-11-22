# Spark集群接入FATE指南

## 一. 概述
fate on spark集群模式

## 二. 部署
需要部署服务包括: FATE、Nginx、RabbitMQ(或Pulsar)、spark+hadoop,可参考以下部署文档:
- [FATE部署指南](../fate_on_eggroll/Fate-allinone_deployment_guide_install.zh.md)
- [Nginx部署指南](nginx_deployment_guide.zh.md)
- [RabbitMQ部署指南](rabbitmq_deployment_guide.zh.md)
- [Pulsar部署指南](pulsar_deployment_guide.zh.md)
- [spark+hadoop部署指南](hadoop_spark_deployment_guide.zh.md)

## 三. 更新FATE配置
fate配置路径:/data/projects/fate/conf
### 1.修改default_engines
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: rabbitmq
  storage: hdfs
```
**注: 若部署的是pulsar， federation可改为"pulsar"**

### 2. 修改Nginx配置
- "conf/service_conf.yaml"
```yaml
fate_on_spark
  nginx:
    host: 127.0.0.1
    http_port: 9300
    grpc_port: 9310
```
**注：请填写实际部署的配置**

### 3. 修改spark和hdfs相关配置
- "conf/service_conf.yaml"
```yaml
fate_on_spark:
  spark:
    home:
    cores_per_node: 40
    nodes: 1
  hdfs:
    name_node: hdfs://fate-cluster
    path_prefix:
```
**注意：请使用实际配置修改上面的spark/home和hdfs/name_node**

### 4.修改rabbitmq或pulsar配置：
####rabbitmq模式：
- "conf/service_conf.yaml"
```yaml
fate_on_spark
  rabbitmq:
    host: 127.0.0.1
    mng_port: 12345
    port: 5672
    user: fate
    password: fate
    route_table:
```
```
相关配置含义
    - host: 主机ip
    - mng_port: 管理端口
    - port: 服务端口
    - user：管理员用户
    - password: 管理员密码
    - route_table: 路由表信息，默认为空
```

- 路由表信息:conf/rabbitmq_route_table.yaml,没有则新建
```yaml
10000:
  host: 127.0.0.1
  port: 5672
9999:
  host: 127.0.0.2
  port: 5672
```

#### pulsar模式:
- "conf/service_conf.yaml"
```yaml
fate_on_spark
  pulsar:
    host: 192.168.0.1
    port: 6650
    mng_port: 8080
    topic_ttl: 5
    # default conf/pulsar_route_table.yaml
    route_table:
```
```
相关配置
    - host: 主机ip
    - port: brokerServicePort
    - mng_port: webServicePort
    - cluster：集群或单机
    - tenant: 合作方需要使用同一个tenant
    - topic_ttl： 回收资源参数
    - route_table: 路由表信息，默认为空
```

- 路由表信息:conf/pulsar_route_table.yaml,没有则新建
```yaml
9999:
  # host can be a domain like 9999.fate.org
  host: 192.168.0.4
  port: 6650
  sslPort: 6651
  # set proxy address for this pulsar cluster
  proxy: ""

10000:
  # host can be a domain like 10000.fate.org
  host: 192.168.0.3
  port: 6650
  sslPort: 6651
  proxy: ""

default:
  # compose host and proxy for party that does not exist in route table
  # in this example, the host for party 8888 will be 8888.fate.org
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
  port: 6650
  sslPort: 6651
```

### 5. spark依赖分发模式
#### 5.1 使用分发模式（推荐）
- "conf/service_conf.yaml"
```yaml
dependent_distribution: true
```

#### 5.1 不使用依赖分发模式（
- "conf/service_conf.yaml"
```yaml
dependent_distribution: false 
```

- 依赖准备:整个fate目录拷贝到每个work节点,目录结构保持一致

- spark配置修改：spark/conf/spark-env.sh
```shell script
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
```



### 6.重启fate flow以使配置生效
```shell script
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fateflow/bin
sh service.sh restart
```

## 四、接入测试
### toy测试
参考文档:[toy测试](../fate_on_eggroll/Fate-allinone_deployment_guide_install.zh.md#61-toy_example部署验证)

## 五、使用fate on spark
### 数据上传

上传引擎使用"HDFS", 配置可参考如下:

```json
{
  "file": "examples/data/breast_hetero_guest.csv",
  "id_delimiter": ",",
  "head": 1,
  "partition": 4,
  "namespace": "experiment",
  "table_name": "breast_hetero_guest",
  "storage_engine": "HDFS"
}

```

```shell script
cd /data/projects/fate/fateflow/
flow data upload -c examples/upload/upload_to_hdfs.json
```

### fate on spark任务提交
- 使用上述方式在guest方和host方各自上传一份hdfs存储类型数据, 并作为reader组件的输入数据发起任务即可

```shell script
cd /data/projects/fate/fateflow/
flow job submit -c  examples/lr/test_hetero_lr_job_conf.json -d examples/lr/test_hetero_lr_job_dsl.json
```


