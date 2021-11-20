## Spark Standalone Access FATE Guide

## I. Overview
FATE already supports spark standalone mode

## II. Deploying
The services to be deployed include: FATE, Nginx, RabbitMQ (or Pulsar), and spark standalone (optional), as described in the following deployment documents:
- [FATE Deployment Guide](../fate_on_eggroll/Fate-allinone_deployment_guide_install.md)
- [Nginx Deployment Guide](nginx_deployment_guide.md)
- [RabbitMQ Deployment Guide](rabbitmq_deployment_guide.md)
- [Pulsar Deployment Guide](pulsar_deployment_guide.md)
- [spark standalone deployment guide](spark_standalone_deployment_guide.md)

### III. Update the FATE configuration
fate configuration path:/data/projects/fate/conf
### 1. Modify default_engines
- "conf/service_conf.yaml"
```yaml
default_engines:
  computing: spark
  federation: rabbitmq
  storage: localfs
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
**Note: Please fill in the actual deployed configuration**


### 3. Modify the rabbitmq or pulsar configuration.
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

### 4. Restart the fate flow to make the configuration take effect
```shell script
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fateflow/bin
sh service.sh restart
```

## IV. Access test
### toy test
Reference document:[toy test](../fate_on_eggroll/Fate-allinone_deployment_guide_install.md#61-verify-toy_example-deployment)

## V. Using fate on spark standalone
### 1. data upload

The upload engine uses "localfs", which can be configured as follows:

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

### 2. spark task submission
- The spark environment:
1. when not deploying spark, the system defaults to using pyspark to submit spark tasks

2. when using the deployed standalone spark, you need to modify the fate configuration: spark_home:

- conf/service_conf.yaml
```yaml
fate_on_spark:
  spark:
    home: /xxx/xxx
```
The system will submit the spark task using spark_home
- Submitting tasks using fate
Use the above method to upload a copy of the localfs store type data on the guest and host side, and use it as input data for the reader component to initiate the task
```shell script
cd /data/projects/fate/fateflow/
flow job submit -c examples/lr/test_hetero_lr_job_conf.json -d examples/lr/test_hetero_lr_job_dsl.json
```
