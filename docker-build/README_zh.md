# 构建FATE的镜像
FATE（Federated AI Technology Enable）是一个工业级的分布式联邦学习框架，它用于在多方在不共享数据的情况下共同训练模型，更多关于FATE的信息请参考[GitHub代码库](https://github.com/FederatedAI/FATE)。
一个完整的FATE系统可划分为若干个部件，有些负责存储、有些负责计算还有一些负责网络发现和数据传输，各个部件各施其责，确保每一次联邦式训练的正常运行，对于部件的划分及其职能请参考[官方文档](https://github.com/FederatedAI/FATE/tree/master/cluster-deploy)。

本文的主要介绍了如何把这些部件构建成Docker镜像，以便与后期的部署和维护。一般来说，使用Docker容器来部署和管理应用有如下好处：
1.	减少了依赖的下载
2.	提高编译的效率
3.	一次构建完毕后就可以进行多次部署

## 前期准备
1.	主机安装Docker  18或以上版本
2.	主机能够访问互联网

## 拉取FATE代码库
  从FATE的Git Hub代码仓库上，用户需要先通过以下命令获取代码：
  
```$ git clone git@github.com:FederatedAI/FATE.git```
  
  拉取完毕后，进入”docker-build/”工作目录。

配置镜像名称
为了减少镜像的容量，我们把镜像划分为以下几类：
- 基础镜像： 安装了必要的依赖包，作为模块镜像的基础镜像(Base Image)。
- 模块镜像： 包含了FATE中某个特定的模块。

用户在开始构建镜像之前需要配置“.env”，通过该文件，镜像在构建完毕后会被打上相应的标签以后续使用，例子如下：
```
  PREFIX=federatedai
  BASE_TAG=1.1-release
  TAG=1.1-release

  # PREFIX: 用于要推送的镜像仓库(Registry)以及其命名空间
  # BASE_TAG: 基础镜像的标签
  # TAG: 模块镜像的标签 
```


## 运行构建镜像的脚本

用户可以使用以下命令来构建镜像：
```$ $ bash build_cluster_docker.sh all```

所有用于构建镜像的“ Dockerfile”文件都存储在“docker/“子目录下。在脚本运行完之后，用户可以通过以下命令来检查构建好的镜像：

```$ docker images | grep federatedai```

一个输出的例子如下：
```
  REPOSITORY                            TAG
  federatedai/egg                       1.1-release
  federatedai/fateboard                 1.1-release
  federatedai/meta-service              1.1-release
  federatedai/python                    1.1-release
  federatedai/roll                      1.1-release
  federatedai/proxy                     1.1-release
  federatedai/federation                1.1-release
  federatedai/base-image                1.1-release
```

## 把镜像推送到镜像仓库（可选）
如果用户需要把构建出来的镜像推送到镜像仓库如DockerHub去的话，需要先通过以下命令登录相应的用户:

```$ docker login username```
   
然后通过脚本把镜像推送到“.env”定义的命名空间中去:

```$ bash build_cluster_docker.sh push```
   
默认情况下脚本会把镜像推送到DockerHub上，".env"中的`PREFIX`字段指定了要把镜像要推送到哪个命名空间上。若用户需要把镜像推送到私有的仓库中，只需要把PREFIX字段修改成相应的值即可。


## 使用离线镜像（可选）
对于一些用户而言，他们的机器可能不允许访问互联网，从而无法下载相应的镜像。此时可以将构建好的镜像打包成一个压缩文件，传输到要部署的机器上之后再把镜像解压出来。
因为FATE的部署需要用到redis和mysql的Docker镜像，因此在构建镜像的机器上没有这两个镜像的话还需要手动拉取。拉取及打包镜像的命令如下:
```
$ docker pull redis
$ docker pull mysql
$ docker save $(docker images | grep -E "redis|mysql" | awk '{print $1":"$2}') -o third-party.images.tar.gz
$ docker save $(docker images | grep federatedai| grep -v -E "base|builder" | awk '{print $1":"$2}') -o fate.images.tar.gz
```

生成"*.images.tar.gz"文件后，需要将其传输到在运行FATE的主机上，运行以下命令导入镜像：
```
$ docker load -i third-party.images.tar.gz
$ docker load -i fate.images.tar.gz
```

部署
Docker镜像生成后可以使用Docker Compose或Kubernetes来部署FATE，部署步骤请参考Kubefate项目，代码仓库地址：https://github.com/FederatedAI/KubeFATE。
