# Spark Cluster Access FATE Guide

## I. Overview
fate on spark cluster model

## II. Deploying
The services to be deployed include: FATE, Nginx, RabbitMQ (or Pulsar), spark+hadoop, which can be found in the following deployment documents:
- [FATE Deployment Guide](../fate_on_eggroll/Fate-allinone_deployment_guide_install.md)
- [Nginx Deployment Guide](common/nginx_deployment_guide.md)
- [RabbitMQ Deployment Guide](rabbitmq_deployment_guide.md)
- [Pulsar Deployment Guide](common/pulsar_deployment_guide.md)
- [spark+hadoop Deployment Guide](common/hadoop_spark_deployment_guide.md)

## III. Update the FATE configuration
fate configuration path:/data/projects/fate/conf
### 1. Modify default_engines
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: rabbitmq
  storage: hdfs
```
**Note: If you are deploying pulsar, federation can be changed to "pulsar "**

### 2. Modify the Nginx configuration
- "conf/service_conf.yaml"
```yaml
fate_on_spark
  nginx:
    host: 127.0.0.1
    http_port: 9300
    grpc_port: 9310
```
**Note: Please fill in the actual deployment configuration**

### 3. modify spark and hdfs related configuration
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
**Note: Please use the actual configuration to modify the above spark/home and hdfs/name_node**

### 4. Modify the rabbitmq or pulsar configuration.
#### rabbitmq mode.
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
Related configuration meanings
    - host: host ip
    - mng_port: management port
    - port: service port
    - user: administrator user
    - password: administrator password
    - route_table: routing table information, default is empty
```

- route_table information: conf/rabbitmq_route_table.yaml, if not, create a new one
```yaml
10000:
  host: 127.0.0.1
  port: 5672
9999:
  host: 127.0.0.2
  port: 5672
```

#### pulsar mode:
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
Related configuration
    - host: host ip
    - port: brokerServicePort
    - mng_port: webServicePort
    - cluster: cluster or single machine
    - tenant: partner needs to use the same tenant
    - topic_ttl: recycling resource parameter
    - route_table: routing table information, default is empty
```

- route_table: conf/pulsar_route_table.yaml, if not, create a new one
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

### 5. spark dependency distribution model
#### 5.1 Using the distribution model (recommended)
- "conf/service_conf.yaml"
```yaml
dependent_distribution: true
```

#### 5.1 Not using the dependent distribution pattern (
- "conf/service_conf.yaml"
```yaml
dependent_distribution: false 
```

- Dependency preparation: copy the entire fate directory to each work node, keeping the directory structure consistent

- spark configuration modification: spark/conf/spark-env.sh
```shell script
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
```



### 6. Restart fate flow to make the configuration take effect
```shell script
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fateflow/bin
sh service.sh restart
```

## IV. Access test
### toy test
Reference document:[toy test](./fate_on_eggroll/Fate-allinone_deployment_guide_install.md#61-verify-toy_example-deployment)

## V. Using fate on spark
### Data upload

The upload engine uses "HDFS", the configuration can be seen below:

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

### fate on spark task submission
- use the above way in the guest side and host side each upload a hdfs storage type data, and as the reader component of the input data to launch the task can be

```shell script
cd /data/projects/fate/fateflow/
flow job submit -c examples/lr/test_hetero_lr_job_conf.json -d examples/lr/test_hetero_lr_job_dsl.json
```
