# FATE单机部署指南

[TOC]

## 1. 说明

服务器配置：

| **数量**      |    1                                                  |
| ------------ | ----------------------------------------------------- |
| **配置**      | 8 core / 16G memory / 500G hard disk                  |
| **操作系统**   | Version: CentOS Linux release 7                       |
| **用户**      | User: app owner:apps                                  |

单机版提供2种部署方式，可以根据实际情况选择：

- 使用Docker镜像安装FATE

- 在主机中安装FATE

## 2. 使用Docker镜像安装FATE（推荐）

建议使用docker镜像，这样可以大大降低遇到问题的可能性

注意，如下示例中的${version}，请用实际的版本号替换，参考[fate.env](../../../fate.env)文件中的FATE版本！

### 2.1 部署前环境检查

- 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。
- 依赖[docker](https://download.docker.com/linux/)，docker建议版本为18.09，您可以使用以下命令验证docker环境：docker --version,docker的起停和其他操作请参考docker --help
- 执行之前，请检查8080是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

### 2.2 拉取镜像

#### 2.2.1 通过公共镜像服务

#### 2.2.2 通过镜像包

   ```bash
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/standalone_fate_docker_image_${version}_release.tar

   docker load < standalone_fate_docker_image_${version}_release.tar

   docker images | grep fate
   ```

   能看到对应${version}的镜像则镜像下载成功

### 2.3 启动

   ```bash
   docker run -d --name fate -p 8080:8080 fate:${version}
   docker ps -a | grep fate
   ```

   能看到对应${version}的容器运行中则启动成功

### 2.4 测试

   - 进入容器

   ```bash
   CONTAINER_ID=`docker ps -aqf "name=fate"`
   docker exec -it ${CONTAINER_ID} bash
   ```

   - [测试项](#4-测试项)

## 3. 在主机中安装FATE

注意，如下示例中的${version}，请用实际的版本号替换，参考[fate.env](../../../fate.env)文件中的FATE版本！

### 3.1 部署前环境检查

本地8080、9360、9380端口是否被占用

   ```bash
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

### 3.2 获取安装包

下载安装包并解压缩

   ```bash
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/1.7.0/release/standalone_fate_install_${version}_release.tar.gz
   tar -xzvf standalone_fate_install_${version}_release.tar.gz
   ```

### 3.3 安装

进入解压后的目录并使用init.sh进行安装

该脚本将自动完成:

- 安装必要的操作系统依赖包
- 安装python36环境
- 安装pypi依赖包
- 安装jdk环境
- 配置FATE环境变量脚本
- 配置fateflow
- 配置fateboard
- 安装fate client

   ```bash
   cd standalone_fate_install_${version}
   sh init.sh init
   ```

### 3.4 启动

   ```bash
   sh init.sh status
   sh init.sh start
   ```

### 3.5 测试

   - 加载环境变量

   ```bash
   source ./bin/init_env.sh
   ```

   - [测试项](#4-测试项)

## 4. 测试项

### 4.1 Toy测试

   ```bash
   flow test toy --guest-party-id 10000 --host-party-id 10000
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   success to calculate secure_sum, it is 2000.0
   ```

### 4.2 单元测试

   ```bash
   bash ./fate/python/federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   there are 0 failed test
   ```

有些用例算法在 [examples](../../../examples/dsl/v2) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://${ip}:8080, ip为`127.0.0.1`或本机实际ip