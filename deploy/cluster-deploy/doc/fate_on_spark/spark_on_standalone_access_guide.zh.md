# Spark单机接入FATE指南

## 一. 概述
FATE已经支持spark单机模式

## 二. 部署
需要部署服务包括: FATE、Nginx、RabbitMQ(或Pulsar)、spark单机(可选),可参考以下部署文档:
- [FATE部署指南](../fate_on_eggroll/Fate-allinone_deployment_guide_install.zh.md)
- [Nginx部署指南](common/nginx_deployment_guide.zh.md)
- [RabbitMQ部署指南](rabbitmq_deployment_guide.zh.md)
- [Pulsar部署指南](common/pulsar_deployment_guide.zh.md)
- [spark单机部署指南](common/spark_standalone_deployment_guide.zh.md)

### 三. 更新FATE配置
fate配置路径:/data/projects/fate/conf
### 1.修改default_engines
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: rabbitmq
  storage: localfs
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


### 3.修改rabbitmq或pulsar配置：
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

### 4.重启fate flow以使配置生效
```shell script
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fateflow/bin
sh service.sh restart
```

## 四、接入测试
### toy测试
参考文档:[toy测试](../fate_on_eggroll/Fate-allinone_deployment_guide_install.zh.md#61-toy_example部署验证)

## 五、使用fate on spark standalone
### 1. 数据上传

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
```shell script
cd /data/projects/fate/fateflow/
flow data upload -c examples/upload/upload_to_localfs.json
```

### 2. spark任务提交
- spark的环境:
1. 不部署spark时,系统默认使用pyspark提交spark任务

2. 使用部署的单机版spark时,需要修改fate配置:spark_home:

- conf/service_conf.yaml
```yaml
fate_on_spark:
  spark:
    home: /xxx/xxx
```
系统会使用spark_home提交spark任务
- 使用fate提交任务
使用上述方式在guest方和host方各自上传一份localfs存储类型数据, 并作为reader组件的输入数据发起任务即可
```shell script
cd /data/projects/fate/fateflow/
flow job submit -c  examples/lr/test_hetero_lr_job_conf.json -d examples/lr/test_hetero_lr_job_dsl.json
```

