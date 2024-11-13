# 使用Docker Compose 部署 FATE

## 前言

[FATE](https://www.fedai.org/ )是一个联邦学习框架，能有效帮助多个机构在满足用户隐私保护、数据安全和政府法规的要求下，进行数据使用和建模。项目地址：（<https://github.com/FederatedAI/FATE/>） 本文档介绍使用Docker Compose部署FATE集群的方法。

## Docker Compose 简介

Compose是用于定义和运行多容器Docker应用程序的工具。通过Compose，您可以使用YAML文件来配置应用程序的服务。然后，使用一个命令，就可以从配置中创建并启动所有服务。要了解有关Compose的所有功能的更多信息，请参阅[相关文档](https://docs.docker.com/compose/#features)。

使用Docker compose 可以方便的部署FATE，下面是使用步骤。

## 目标

两个可以互通的FATE实例，每个实例均包括FATE所有组件。

## 准备工作

1. 两个主机（物理机或者虚拟机，都是Centos7系统）；
2. 所有主机安装Docker 版本 :  19.03.0+；
3. 所有主机安装Docker Compose 版本: 1.27.0+；
4. 部署机可以联网，所以主机相互之间可以网络互通；
5. 运行机已经下载FATE的各组件镜像，如果无法连接dockerhub，请考虑使用harbor（[Harbor 作为本地镜像源](../registry/README.md)）或者使用离线部署（离线构建镜像参考文档[构建镜像]( https://github.com/FederatedAI/FATE-Builder/tree/main/docker-build)）。
6. 运行FATE的主机推荐配置8CPUs和16G RAM。

### 下载部署脚本

在任意机器上下载合适的KubeFATE版本，可参考 [releases pages](https://github.com/FederatedAI/KubeFATE/releases)，然后解压。

### 修改镜像配置文件（可选）

在默认情况下，脚本在部署期间会从 [Docker Hub](https://hub.docker.com/search?q=federatedai&type=image)中下载镜像。

对于中国的用户可以用使用国内镜像源：
具体方法是通过编辑docker-deploy目录下的.env文件,给`RegistryURI`参数填入以下字段

```bash
RegistryURI=hub.c.163.com
```

如果在运行机器上已经下载或导入了所需镜像，部署将会变得非常容易。

### 手动下载镜像（可选）

如果运行机没有FATE组件的镜像，可以通过以下命令从Docker Hub获取镜像。FATE镜像的版本`<version>`可在[release页面](https://github.com/FederatedAI/FATE/releases)上查看，其中serving镜像的版本信息在[这个页面](https://github.com/FederatedAI/FATE-Serving/releases)：

```bash
docker pull federatedai/eggroll:<version>-release
docker pull federatedai/fateboard:<version>-release
docker pull federatedai/fateflow:<version>-release
docker pull federatedai/serving-server:<version>-release
docker pull federatedai/serving-proxy:<version>-release
docker pull federatedai/serving-admin:<version>-release
docker pull bitnami/zookeeper:3.7.0 
docker pull mysql:8.0.28
```

检查所有镜像是否下载成功。

```bash
$ docker images
REPOSITORY                         TAG 
federatedai/eggroll                <version>-release
federatedai/fateboard              <version>-release
federatedai/fateflow               <version>-release
federatedai/client                 <version>-release
federatedai/serving-server         <version>-release
federatedai/serving-proxy          <version>-release
federatedai/serving-admin          <version>-release
bitnami/zookeeper                  3.7.0 
mysql                              8.0.28
```

### 离线部署（可选）

当我们的运行机器处于无法连接外部网络的时候，就无法从Docker Hub下载镜像，建议使用[Harbor](https://goharbor.io/)作为本地镜像仓库。安装Harbor请参考[文档](https://github.com/FederatedAI/KubeFATE/blob/master/registry/install_harbor.md)。在`.env`文件中，将`RegistryURI`变量更改为Harbor的IP。如下面 192.168.10.1是Harbor IP的示例。

```bash
$ cd KubeFATE/
$ vi .env

...
RegistryURI=192.168.10.1/federatedai
...
```

## 用Docker Compose部署FATE

  ***如果在之前你已经部署过其他版本的FATE，请删除清理之后再部署新的版本，[删除部署](#删除部署).***

### 配置需要部署的实例数目

部署脚本提供了部署多个FATE实例的功能，下面的例子我们部署在两个机器上，每个机器运行一个FATE实例，这里两台机器的IP分别为*192.168.7.1*和*192.168.7.2*

根据需求修改配置文件`kubeFATE\docker-deploy\parties.conf`。

`parties.conf`配置文件配置项的含义查看这个文档[parties.conf文件介绍](../docs/configurations/Docker_compose_Partys_configuration.md)

下面是修改好的文件，`party 10000`的集群将部署在*192.168.7.1*上，而`party 9999`的集群将部署在*192.168.7.2*上。

```bash
user=fate
dir=/data/projects/fate
party_list=(10000 9999)
party_ip_list=(192.168.7.1 192.168.7.2)
serving_ip_list=(192.168.7.1 192.168.7.2)

computing=Eggroll
federation=Eggroll
storage=Eggroll

algorithm=Basic
device=IPCL

compute_core=4

......

```

* 使用Spark+Rabbitmq的部署方式的文档可以参考[这里](../docs/FATE_On_Spark.md).
* 使用Spark+Pulsar的部署方式的文档可以参考[这里](../docs/FATE_On_Spark_With_Pulsar.md).
* 使用Spark+local Pulsar的部署方式的文档可以参考[这里](TBD)

使用Docker-compose部署FATE可以支持多种种不同的类型引擎的组合(对computing federation storage的选择)，关于不同类型的FATE的更多细节查看: [不同类型FATE的架构介绍](../docs/Introduction_to_Engine_Architecture_zh.md)。

`algorithm`和`device`的配置可以查看这里[FATE_Algorithm_and_Computational_Acceleration_Selection.md](../docs/FATE_Algorithm_and_Computational_Acceleration_Selection.md)

**注意**: 默认情况下不会部署exchange组件。如需部署，用户可以把服务器IP填入上述配置文件的`exchangeip`中，该组件的默认监听端口为9371。

在运行部署脚本之前，需要确保部署机器可以ssh免密登录到两个运行节点主机上。user代表免密的用户。

在运行FATE的主机上，user是非root用户的，需要有`/data/projects/fate`文件夹权限和docker权限。如果是root用户则不需要任何其他操作。

```bash
# 创建一个组为docker的fate用户
[user@localhost]$ sudo useradd -s /bin/bash -g docker -d /home/fate fate
# 设置用户密码
[user@localhost]$ sudo passwd fate
# 创建docker-compose部署目录
[user@localhost]$ sudo mkdir -p /data/projects/fate /home/fate
# 修改docker-compose部署目录对应用户和组
[user@localhost]$ sudo chown -R fate:docker /data/projects/fate /home/fate
# 选择用户
[user@localhost]$ sudo su fate
# 查看是否拥有docker权限
[fate@localhost]$ docker ps
CONTAINER ID  IMAGE   COMMAND   CREATED   STATUS    PORTS   NAMES
# 查看docker-compose部署目录
[fate@localhost]$ ls -l /data/projects/
total 0
drwxr-xr-x. 2 fate docker 6 May 27 00:51 fate
```

### GPU支持

从v1.11.1开始docker compose部署支持使用GPU的FATE部署，如果要使用GPU，你需要先搞定GPU的docker环境。可以参考docker的官方文档（<https://docs.docker.com/config/containers/resource_constraints/#gpu>）。

要使用GPU需要修改配置,这两个都需要修改

```sh
algorithm=NN
device=GPU

gpu_count=1
```

FATE GPU的使用只有fateflow组件，所以每个Party最少需要有一个GPU。

*gpu_count会映射为count，参考 [Docker compose GPU support](https://docs.docker.com/compose/gpu-support/)*

### 执行部署脚本

**注意：**在运行以下命令之前，所有目标主机必须

* 允许使用 SSH 密钥进行无密码 SSH 访问（否则我们将需要为每个主机多次输入密码）。
* 满足 [准备工作](#准备工作) 中指定的要求。

要将 FATE 部署到所有已配置的目标主机，请使用以下命令：

以下修改可在任意机器执行。

进入目录`kubeFATE\docker-deploy`，然后运行：

```bash
bash ./generate_config.sh          # 生成部署文件
```

脚本将会生成10000、9999两个组织(Party)的部署文件，然后打包成tar文件。接着把tar文件`confs-<party-id>.tar`、`serving-<party-id>.tar`分别复制到party对应的主机上并解包，解包后的文件默认在`/data/projects/fate`目录下。然后脚本将远程登录到这些主机并使用docker compose命令启动FATE实例。

默认情况下，脚本会同时启动训练和服务集群。 如果您需要单独启动它们，请将 `--training` 或 `--serving` 添加到 `docker_deploy.sh` 中，如下所示。

（可选）要部署各方训练集群，请使用以下命令：

```bash
bash ./docker_deploy.sh all --training
```

（可选）要部署各方服务集群，请使用以下命令：

```bash
bash ./docker_deploy.sh all --serving
```

（可选）要将 FATE 部署到单个目标主机，请使用以下命令和参与方的 ID（下例中为 10000）：

```bash
bash ./docker_deploy.sh 10000
```

（可选）要将交换节点部署到目标主机，请使用以下命令：

```bash
bash ./docker_deploy.sh exchange
```

命令完成后，登录到任何主机并使用 `docker compose ps` 来验证集群的状态。 示例输出如下：

```bash
ssh fate@192.168.7.1
```

使用以下命令验证实例状态，

```bash
cd /data/projects/fate/confs-10000
docker compose ps
```

输出显示如下，若各个组件状态都是`Up`状态，并且fateflow的状态还是(healthy)，说明部署成功。

```bash
NAME                           IMAGE                                  COMMAND                  SERVICE             CREATED              STATUS                        PORTS
confs-10000-client-1           federatedai/client:2.0.0-release      "bash -c 'pipeline i…"   client              About a minute ago   Up About a minute             0.0.0.0:20000->20000/tcp, :::20000->20000/tcp
confs-10000-clustermanager-1   federatedai/eggroll:2.0.0-release     "/tini -- bash -c 'j…"   clustermanager      About a minute ago   Up About a minute             4670/tcp
confs-10000-fateboard-1        federatedai/fateboard:2.0.0-release   "/bin/sh -c 'java -D…"   fateboard           About a minute ago   Up About a minute             0.0.0.0:8080->8080/tcp, :::8080->8080/tcp
confs-10000-fateflow-1         federatedai/fateflow:2.0.0-release    "/bin/bash -c 'set -…"   fateflow            About a minute ago   Up About a minute (healthy)   0.0.0.0:9360->9360/tcp, :::9360->9360/tcp, 0.0.0.0:9380->9380/tcp, :::9380->9380/tcp
confs-10000-mysql-1            mysql:8.0.28                           "docker-entrypoint.s…"   mysql               About a minute ago   Up About a minute             3306/tcp, 33060/tcp
confs-10000-nodemanager-1      federatedai/eggroll:2.0.0-release     "/tini -- bash -c 'j…"   nodemanager         About a minute ago   Up About a minute             4671/tcp
confs-10000-osx-1         federatedai/osx:2.0.0-release     "/tini -- bash -c 'j…"   osx            About a minute ago   Up About a minute             0.0.0.0:9370->9370/tcp, :::9370->9370/tcp
```

### 验证部署

docker-compose上的FATE启动成功之后需要验证各个服务是否都正常运行，我们可以通过验证toy_example示例来检测。

选择192.168.7.1这个节点验证，使用以下命令验证：

```bash
# 在192.168.7.1上执行下列命令

# 进入client组件容器内部
$ docker compose exec client bash
# toy 验证
$ flow test toy --guest-party-id 10000 --host-party-id 9999        
```

如果测试通过，屏幕将显示类似如下消息：

```bash
"2019-08-29 07:21:25,353 - secure_add_guest.py[line:96] - INFO: begin to init parameters of secure add example guest"
"2019-08-29 07:21:25,354 - secure_add_guest.py[line:99] - INFO: begin to make guest data"
"2019-08-29 07:21:26,225 - secure_add_guest.py[line:102] - INFO: split data into two random parts"
"2019-08-29 07:21:29,140 - secure_add_guest.py[line:105] - INFO: share one random part data to host"
"2019-08-29 07:21:29,237 - secure_add_guest.py[line:108] - INFO: get share of one random part data from host"
"2019-08-29 07:21:33,073 - secure_add_guest.py[line:111] - INFO: begin to get sum of guest and host"
"2019-08-29 07:21:33,920 - secure_add_guest.py[line:114] - INFO: receive host sum from guest"
"2019-08-29 07:21:34,118 - secure_add_guest.py[line:121] - INFO: success to calculate secure_sum, it is 2000.0000000000002"
```

### 验证Serving-Service功能

#### Host方操作

##### 进入party10000 client容器

```bash
cd /data/projects/fate/confs-10000
docker compose exec client bash
```

##### 上传host数据

```bash
flow data upload -c fateflow/examples/upload/upload_host.json
```

#### Guest方操作

##### 进入party9999 client容器

```bash
cd /data/projects/fate/confs-9999
docker compose exec client bash
```

##### 上传guest数据

```bash
flow data upload -c fateflow/examples/upload/upload_guest.json
```

##### 提交任务

```bash
flow job submit -d fateflow/examples/lr/test_hetero_lr_job_dsl.json -c fateflow/examples/lr/test_hetero_lr_job_conf.json
```

output：

```json
{
    "data": {
        "board_url": "http://fateboard:8080/index.html#/dashboard?job_id=202111230933232084530&role=guest&party_id=9999",
        "code": 0,
        "dsl_path": "/data/projects/fate/fate_flow/jobs/202111230933232084530/job_dsl.json",
        "job_id": "202111230933232084530",
        "logs_directory": "/data/projects/fate/fate_flow/logs/202111230933232084530",
        "message": "success",
        "model_info": {
            "model_id": "arbiter-10000#guest-9999#host-10000#model",
            "model_version": "202111230933232084530"
        },
        "pipeline_dsl_path": "/data/projects/fate/fate_flow/jobs/202111230933232084530/pipeline_dsl.json",
        "runtime_conf_on_party_path": "/data/projects/fate/fate_flow/jobs/202111230933232084530/guest/9999/job_runtime_on_party_conf.json",
        "runtime_conf_path": "/data/projects/fate/fate_flow/jobs/202111230933232084530/job_runtime_conf.json",
        "train_runtime_conf_path": "/data/projects/fate/fate_flow/jobs/202111230933232084530/train_runtime_conf.json"
    },
    "jobId": "202111230933232084530",
    "retcode": 0,
    "retmsg": "success"
}
```

##### 查看训练任务状态

```bash
flow task query -r guest -j 202111230933232084530 | grep -w f_status
```

output:

```bash
            "f_status": "success",
            "f_status": "waiting",
            "f_status": "running",
            "f_status": "waiting",
            "f_status": "waiting",
            "f_status": "success",
            "f_status": "success",
```

等到所有的`waiting`状态变为`success`.

##### 部署模型

```bash
flow model deploy --model-id arbiter-10000#guest-9999#host-10000#model --model-version 202111230933232084530
```

```json
{
    "data": {
        "arbiter": {
            "10000": 0
        },
        "detail": {
            "arbiter": {
                "10000": {
                    "retcode": 0,
                    "retmsg": "deploy model of role arbiter 10000 success"
                }
            },
            "guest": {
                "9999": {
                    "retcode": 0,
                    "retmsg": "deploy model of role guest 9999 success"
                }
            },
            "host": {
                "10000": {
                    "retcode": 0,
                    "retmsg": "deploy model of role host 10000 success"
                }
            }
        },
        "guest": {
            "9999": 0
        },
        "host": {
            "10000": 0
        },
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202111230954255210490"
    },
    "retcode": 0,
    "retmsg": "success"
}
```

*后面需要用到的`model_version`都是这一步得到的`"model_version": "202111230954255210490"`*

##### 修改加载模型的配置

```bash
cat > fateflow/examples/model/publish_load_model.json <<EOF
{
  "initiator": {
    "party_id": "9999",
    "role": "guest"
  },
  "role": {
    "guest": [
      "9999"
    ],
    "host": [
      "10000"
    ],
    "arbiter": [
      "10000"
    ]
  },
  "job_parameters": {
    "model_id": "arbiter-10000#guest-9999#host-10000#model",
    "model_version": "202111230954255210490"
  }
}
EOF
```

##### 加载模型

```bash
flow model load -c fateflow/examples/model/publish_load_model.json
```

output:

```json
{
    "data": {
        "detail": {
            "guest": {
                "9999": {
                    "retcode": 0,
                    "retmsg": "success"
                }
            },
            "host": {
                "10000": {
                    "retcode": 0,
                    "retmsg": "success"
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
    "jobId": "202111240844337394000",
    "retcode": 0,
    "retmsg": "success"
}
```

##### 修改绑定模型的配置

```bash
cat > fateflow/examples/model/bind_model_service.json <<EOF
{
    "service_id": "test",
    "initiator": {
        "party_id": "9999",
        "role": "guest"
    },
    "role": {
        "guest": ["9999"],
        "host": ["10000"],
        "arbiter": ["10000"]
    },
    "job_parameters": {
        "work_mode": 1,
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202111230954255210490"
    }
}
EOF
```

##### 绑定模型

```bash
flow model bind -c fateflow/examples/model/bind_model_service.json
```

output:

```json
{
    "retcode": 0,
    "retmsg": "service id is test"
}
```

##### 在线测试

发送以下信息到"GUEST"方的推理服务"{SERVING_SERVICE_IP}:8059/federation/v1/inference"

```bash
$ curl -X POST -H 'Content-Type: application/json' -i 'http://192.168.7.2:8059/federation/v1/inference' --data '{
  "head": {
    "serviceId": "test"
  },
  "body": {
    "featureData": {
        "x0": 1.88669,
        "x1": -1.359293,
        "x2": 2.303601,
        "x3": 2.00137,
        "x4": 1.307686
    },
    "sendToRemoteFeatureData": {
        "phone_num": "122222222"
    }
  }
}'
```

output:

```json
{"retcode":0,"retmsg":"","data":{"score":0.018025086161221948,"modelId":"guest#9999#arbiter-10000#guest-9999#host-10000#model","modelVersion":"202111240318516571130","timestamp":1637743473990},"flag":0}
```

### 删除部署

在部署机器上运行以下命令可以停止所有FATE集群：

```bash
bash ./docker_deploy.sh --delete all
```

如果想要彻底删除在运行机器上部署的FATE，可以分别登录节点，然后运行命令：

```bash
cd /data/projects/fate/confs-<id>/  # <id> 组织的id，本例中代表10000或者9999
docker-compose down
rm -rf ../confs-<id>/               # 删除docker-compose部署文件
```

### 可能遇到的问题

#### 采用docker hub下载镜像速度可能较慢

解决办法：可以自己构建镜像，自己构建镜像参考[这里](https://github.com/FederatedAI/FATE/tree/master/docker-build)。

#### 运行脚本`./docker_deploy.sh all`的时候提示需要输入密码

解决办法：检查免密登陆是否正常。ps:直接输入对应主机的用户密码也可以继续运行。

#### CPU指令集问题

解决办法：查看[wiki](https://github.com/FederatedAI/KubeFATE/wiki/KubeFATE)页面的storage-service部分
