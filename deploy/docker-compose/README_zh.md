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
docker pull federatedai/eggroll:3.2.0-release
docker pull federatedai/fateflow:2.2.0-release
docker pull federatedai/osx:2.2.0-release
docker pull federatedai/fateboard:2.1.1-release
docker pull mysql:8.0.28
```

检查所有镜像是否下载成功。

```bash
$ docker images
REPOSITORY                         TAG 
federatedai/fateflow         2.2.0-release
federatedai/eggroll          3.2.0-release
federatedai/osx              2.2.0-release
federatedai/fateboard        2.1.1-release
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

# Engines:
# Computing : Eggroll, Spark, Spark_local
computing=Eggroll
# Federation: OSX(computing: Eggroll/Spark/Spark_local), Pulsar/RabbitMQ(computing: Spark/Spark_local)
federation=OSX
# Storage: Eggroll(computing: Eggroll), HDFS(computing: Spark), LocalFS(computing: Spark_local)
storage=Eggroll
# Algorithm: Basic, NN, ALL
algorithm=Basic
# Device: CPU, IPCL, GPU
device=CPU
   
# spark and eggroll 
compute_core=16
   
# You only need to configure this parameter when you want to use the GPU, the default value is 1
gpu_count=0
   
# modify if you are going to use an external db
mysql_ip=mysql
mysql_user=fate
mysql_password=fate_dev
mysql_db=fate_flow
serverTimezone=UTC
   
name_node=hdfs://namenode:9000
   
# Define fateboard login information
fateboard_username=admin
fateboard_password=admin

```

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

脚本将会生成10000、9999两个组织(Party)的部署文件，然后打包成tar文件。接着把tar文件`confs-<party-id>.tar`复制到party对应的主机上并解包，解包后的文件默认在`/data/projects/fate`目录下。然后脚本将远程登录到这些主机并使用docker compose命令启动FATE实例。

默认情况下，脚本会同时启动训练和服务集群。 如果您需要单独启动它们，请将 `--training` 添加到 `docker_deploy.sh` 中，如下所示。

（可选）要部署各方训练集群，请使用以下命令：

```bash
bash ./docker_deploy.sh all --training
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
docker-compose ps
```

输出显示如下，若各个组件状态都是`Up`状态，并且fateflow的状态还是(healthy)，说明部署成功。

```bash
NAME                           IMAGE                                  COMMAND                  SERVICE             CREATED              STATUS                        PORTS
confs-10000-clustermanager-1   federatedai/eggroll:3.2.0-release     "/tini -- bash -c 'j…"   clustermanager      About a minute ago   Up About a minute             4670/tcp
confs-10000-fateflow-1         federatedai/fateflow:2.2.0-release    "/bin/bash -c 'set -…"   fateflow            About a minute ago   Up About a minute (healthy)   192.168.7.1:9360->9360/tcp, :::9360->9360/tcp, 192.168.7.1:9380->9380/tcp, :::9380->9380/tcp
confs-10000-mysql-1            mysql:8.0.28                          "docker-entrypoint.s…"   mysql               About a minute ago   Up About a minute             3306/tcp, 33060/tcp
confs-10000-nodemanager-1      federatedai/eggroll:3.2.0-release     "/tini -- bash -c 'j…"   nodemanager         About a minute ago   Up About a minute             4671/tcp
confs-10000-osx-1              federatedai/osx:2.2.0-release         "/tini -- bash -c 'j…"   osx                 About a minute ago   Up About a minute             192.168.7.1:9370->9370/tcp, :::9370->9370/tcp
confs-10000-fateboard-1        federatedai/fateboard:2.1.1-release   "sh -c 'java -Dsprin…"   fateboard           About a minute ago   Up About a minute             192.168.7.1:8080->8080/tcp
```

### 验证部署

docker-compose上的FATE启动成功之后需要验证各个服务是否都正常运行，我们可以通过验证toy_example示例来检测。

选择192.168.7.1这个节点验证，使用以下命令验证：

```bash
# 在192.168.7.1上执行下列命令

# 进入fateflow组件容器内部
$ docker-compose exec fateflow bash
# toy 验证
$ flow test toy --guest-party-id 10000 --host-party-id 9999        
```

如果测试通过，屏幕将显示类似如下消息：

```bash
toy test job xxxxx is success
```

### 上传数据，发起任务

#### Host方操作

##### 进入party10000 fateflow容器

```bash
cd /data/projects/fate/confs-10000
docker-compose exec fateflow bash
```

##### 上传host数据
执行python脚本，上传数据
```bash
# 上传数据（单边的， 双边需要在另一方再次执行）
from fate_client.pipeline import FateFlowPipeline
   
guest_data_path="/data/projects/fate/examples/data/breast_hetero_guest.csv"
host_data_path="/data/projects/fate/examples/data/breast_hetero_host.csv"
   
data_pipeline = FateFlowPipeline().set_parties(local="0")
guest_meta = {
       "delimiter": ",", "dtype": "float64", "label_type": "int64","label_name": "y", "match_id_name": "id"
   }
host_meta = {
       "delimiter": ",", "input_format": "dense", "match_id_name": "id"
   }
data_pipeline.transform_local_file_to_dataframe(file=guest_data_path, namespace="experiment", name="breast_hetero_guest",
                                                   meta=guest_meta, head=True, extend_sid=True)
data_pipeline.transform_local_file_to_dataframe(file=host_data_path, namespace="experiment", name="breast_hetero_host",
                                                   meta=host_meta, head=True, extend_sid=True)
```

#### Guest方操作

##### 进入party9999 fateflow容器

```bash
cd /data/projects/fate/confs-9999
docker-compose exec fateflow bash
```

##### 上传guest数据
执行python脚本，上传数据
```bash
# 上传数据（单边的， 双边需要在另一方再次执行）
from fate_client.pipeline import FateFlowPipeline
   
guest_data_path="/data/projects/fate/examples/data/breast_hetero_guest.csv"
host_data_path="/data/projects/fate/examples/data/breast_hetero_host.csv"
   
data_pipeline = FateFlowPipeline().set_parties(local="0")
guest_meta = {
       "delimiter": ",", "dtype": "float64", "label_type": "int64","label_name": "y", "match_id_name": "id"
   }
host_meta = {
       "delimiter": ",", "input_format": "dense", "match_id_name": "id"
   }
data_pipeline.transform_local_file_to_dataframe(file=guest_data_path, namespace="experiment", name="breast_hetero_guest",
                                                   meta=guest_meta, head=True, extend_sid=True)
data_pipeline.transform_local_file_to_dataframe(file=host_data_path, namespace="experiment", name="breast_hetero_host",
                                                   meta=host_meta, head=True, extend_sid=True)
```

##### 提交任务
执行python脚本，发起任务
```bash
# 发起任务
from fate_client.pipeline.components.fate import (
       HeteroSecureBoost,
       Reader,
       PSI,
       Evaluation
   )
from fate_client.pipeline import FateFlowPipeline
   
   
# create pipeline for training
pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")
   
# create reader task_desc
reader_0 = Reader("reader_0")
reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
   
# create psi component_desc
psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])
   
# create hetero secure_boost component_desc
hetero_secureboost_0 = HeteroSecureBoost(
       "hetero_secureboost_0", num_trees=1, max_depth=5,
       train_data=psi_0.outputs["output_data"],
       validate_data=psi_0.outputs["output_data"]
   )
   
# create evaluation component_desc
evaluation_0 = Evaluation(
       'evaluation_0', runtime_parties=dict(guest="9999"), metrics=["auc"], input_datas=[hetero_secureboost_0.outputs["train_output_data"]]
   )
   
# add training task
pipeline.add_tasks([reader_0, psi_0, hetero_secureboost_0, evaluation_0])
   
# compile and train
pipeline.compile()
pipeline.fit()
   
# print metric and model info
print (pipeline.get_task_info("hetero_secureboost_0").get_output_model())
print (pipeline.get_task_info("evaluation_0").get_output_metric())
   
# deploy task for inference
pipeline.deploy([psi_0, hetero_secureboost_0])
   
# create pipeline for predicting
predict_pipeline = FateFlowPipeline()
   
# add input to deployed_pipeline
deployed_pipeline = pipeline.get_deployed_pipeline()
reader_1 = Reader("reader_1")
reader_1.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_1.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]
   
# add task to predict pipeline
predict_pipeline.add_tasks([reader_1, deployed_pipeline])
   
# compile and predict
predict_pipeline.compile()
predict_pipeline.predict()
```


任务成功后，屏幕将显示下方类似结果
output:

```bash
Job is success!!! Job id is 202404031636558952240, response_data={'apply_resource_time': 1712133417129, 'cores': 4, 'create_time': 1712133415928, 'dag': {'dag': {'conf': {'auto_retries': 0, 'computing_partitions': 8, 'cores': None, 'extra': None, 'inheritance': None, 'initiator_party_id': '9999', 'model_id': '202404031636558952240', 'model_version': '0', 'model_warehouse': {'model_id': '202404031635272687860', 'model_version': '0'}, 'priority': None, 'scheduler_party_id': '9999', 'sync_type': 'callback', 'task': None}, 'parties': [{'party_id': ['9999'], 'role': 'guest'}, {'party_id': ['10000'], 'role': 'host'}], 'party_tasks': {'guest_9999': {'conf': {}, 'parties': [{'party_id': ['9999'], 'role': 'guest'}], 'tasks': {'reader_1': {'conf': None, 'parameters': {'name': 'breast_hetero_guest', 'namespace': 'experiment'}}}}, 'host_10000': {'conf': {}, 'parties': [{'party_id': ['10000'], 'role': 'host'}], 'tasks': {'reader_1': {'conf': None, 'parameters': {'name': 'breast_hetero_host', 'namespace': 'experiment'}}}}}, 'stage': 'predict', 'tasks': {'hetero_secureboost_0': {'component_ref': 'hetero_secureboost', 'conf': None, 'dependent_tasks': ['psi_0'], 'inputs': {'data': {'test_data': {'task_output_artifact': [{'output_artifact_key': 'output_data', 'output_artifact_type_alias': None, 'parties': [{'party_id': ['9999'], 'role': 'guest'}, {'party_id': ['10000'], 'role': 'host'}], 'producer_task': 'psi_0'}]}}, 'model': {'input_model': {'model_warehouse': {'output_artifact_key': 'output_model', 'output_artifact_type_alias': None, 'parties': [{'party_id': ['9999'], 'role': 'guest'}, {'party_id': ['10000'], 'role': 'host'}], 'producer_task': 'hetero_secureboost_0'}}}}, 'outputs': None, 'parameters': {'gh_pack': True, 'goss': False, 'goss_start_iter': 0, 'hist_sub': True, 'l1': 0, 'l2': 0.1, 'learning_rate': 0.3, 'max_bin': 32, 'max_depth': 5, 'min_child_weight': 1, 'min_impurity_split': 0.01, 'min_leaf_node': 1, 'min_sample_split': 2, 'num_class': 2, 'num_trees': 1, 'objective': 'binary:bce', 'other_rate': 0.1, 'split_info_pack': True, 'top_rate': 0.2}, 'parties': None, 'stage': None}, 'psi_0': {'component_ref': 'psi', 'conf': None, 'dependent_tasks': ['reader_1'], 'inputs': {'data': {'input_data': {'task_output_artifact': {'output_artifact_key': 'output_data', 'output_artifact_type_alias': None, 'parties': [{'party_id': ['9999'], 'role': 'guest'}, {'party_id': ['10000'], 'role': 'host'}], 'producer_task': 'reader_1'}}}, 'model': None}, 'outputs': None, 'parameters': {}, 'parties': None, 'stage': 'default'}, 'reader_1': {'component_ref': 'reader', 'conf': None, 'dependent_tasks': None, 'inputs': None, 'outputs': None, 'parameters': {}, 'parties': None, 'stage': 'default'}}}, 'kind': 'fate', 'schema_version': '2.1.0'}, 'description': '', 'elapsed': 62958, 'end_time': 1712133480145, 'engine_name': 'eggroll', 'flow_id': '', 'inheritance': {}, 'initiator_party_id': '9999', 'job_id': '202404031636558952240', 'memory': 0, 'model_id': '202404031636558952240', 'model_version': '0', 'parties': [{'party_id': ['9999'], 'role': 'guest'}, {'party_id': ['10000'], 'role': 'host'}], 'party_id': '9999', 'progress': 100, 'protocol': 'fate', 'remaining_cores': 4, 'remaining_memory': 0, 'resource_in_use': False, 'return_resource_time': 1712133480016, 'role': 'guest', 'scheduler_party_id': '9999', 'start_time': 1712133417187, 'status': 'success', 'status_code': None, 'tag': 'job_end', 'update_time': 1712133480145, 'user_name': ''}
Total time: 0:01:04
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

解决办法：查看[wiki](https://github.com/FederatedAI/KubeFATE/wiki/KubeFATE)页面的storage-service部分。
