# FATE on Spark local部署指南

## 概述
FATE在1.7版本中支持spark单机模式

## 部署与配置
### 部署
需要安装的软件包括: rabbitmq || pulsar(二选一), Nginx

具体部署可参考:
[rabbitmq部署指南](https://github.com/FederatedAI/FATE/blob/master/cluster-deploy/doc/fate_on_spark/rabbitmq_deployment_guide_zh.md)
[pulsar部署指南](https://github.com/FederatedAI/FATE/blob/master/cluster-deploy/doc/fate_on_spark/pulsar_deployment_guide_zh.md).
[Nginx部署指南](https://github.com/FederatedAI/FATE/blob/master/cluster-deploy/doc/fate_on_spark/fate_deployment_step_by_step_zh.md#26-%E9%83%A8%E7%BD%B2nginx)

### 更新FATE配置
fate需要更新的配置包括两项(rabbitmq和pulsar二选一)：

**rabbitmq模式**:
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: rabbitmq
  storage: localfs
fate_on_spark
  rabbitmq:
    host: 127.0.0.1
    mng_port: 12345
    port: 5672
    user: fate
    password: fate
    route_table:
  nginx:
    host: 127.0.0.1
    http_port: 9300
    grpc_port: 9310
```
- "conf/rabbitmq_route_table.yaml"
```yaml
10000:
  host: 127.0.0.1
  port: 5672
9999:
  host: 127.0.0.2
  port: 5672
```

**pulsar模式**:
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: pulsar
  storage: localfs
fate_on_spark
  pulsar:
    host: 192.168.0.1
    port: 6650
    mng_port: 8080
    topic_ttl: 5
    # default conf/pulsar_route_table.yaml
    route_table:
  nginx:
    host: 127.0.0.1
    http_port: 9300
    grpc_port: 9310
```
- "conf/pulsar_route_table.yaml"
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

### 数据上传

上传引擎使用"localfs", 配置可参考如下:

```json
{
  "file": "examples/data/breast_hetero_guest.csv",
  "id_delimiter": ",",
  "head": 1,
  "partition": 4,
  "namespace": "experiment",
  "table_name": "breast_hetero_guest",
  "storage_engine": "LOCALFS"
}

```

### spark任务提交
1. 不部署spark
系统默认使用pyspark提交spark任务

2. 部署单机版spark,需要修改spark_home:
- conf/service_conf.yaml
```yaml
fate_on_spark:
  spark:
    home: /xxx/xxx
    cores_per_node: 20
    nodes: 12
```
系统会使用spark_home提交任务
