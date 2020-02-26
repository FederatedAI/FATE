## ****FATE单机部署指南****

单机版提供三种部署方式，可以根据实际情况选择：

- 使用Docker镜像安装FATE（推荐）

- 在主机中安装FATE

- 使用Docker从源代码构建FATE（需要40分钟或更长时间）

  

#### 1) 使用Docker镜像安装FATE（推荐） 

建议使用docker镜像，这样可以大大降低遇到问题的可能性。

1. 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。

2. 依赖[docker](https://download.docker.com/linux/)和[docker-compose](https://github.com/docker/compose/releases/tag/1.24.0)，docker建议版本为18.09，docker-compose建议版本为1.24.0，您可以使用以下命令验证docker环境：docker --version和docker-compose --version，docker的起停和其他操作请参考docker --help。

3. 执行之前，请检查8080、9060和9080端口是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

   请按照以下步骤操作:


```
#获取安装包
FATE $ wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone-fate-1.3.0.tar.gz
FATE $tar -xvf docker_standalone-fate-1.3.0.tar.gz

#执行部署
FATE $ cd docker_standalone-fate-1.3.0
FATE $ bash install_standalone_docker.sh

#验证和测试
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh

```

有些用例算法在 [examples](../../examples/federatedml-1.x-examples) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080。



#### 2) 在主机中安装FATE

1. 检查本地8080、9360、9380端口是否被占用。

   ```
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

2. 下载独立版本的压缩包并解压缩。

   ```
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/standalone-fate-master-1.3.0.tar.gz
   tar -xvf  standalone-fate-master-1.3.0.tar.gz
   ```

3. 进入FATE目录并执行init.sh.

   ```
   cd standalone-fate-master-1.3.0
   source init.sh init
   ```

4. 执行测试.

   ```
   cd standalone-fate-master-1.3.0
   bash ./federatedml/test/run_test.sh
   ```

有些用例算法在 [examples](../../examples/federatedml-1.x-examples) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080。



#### 3) 使用Docker从源代码构建FATE

1. 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。

2. 依赖[docker](https://download.docker.com/linux/)和[docker-compose](https://github.com/docker/compose/releases/tag/1.24.0)，docker建议版本为18.09，docker-compose建议版本为1.24.0，您可以使用以下命令验证docker环境：docker --version和docker-compose --version，docker的起停和其他操作请参考docker --help。

3. 执行之前，请检查8080、9060和9080端口是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

4. 大约需要40分钟才能执行完成，请耐心等待。

   请按照以下步骤操作:

```
#获取安装包
FATE $ git clone https://github.com/FederatedAI/FATE.git

#执行部署
FATE $ cd FATE/standalone-deploy
FATE $ bash build_standalone_docker.sh init

#验证和测试
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh

```

有些用例算法在 [examples](../../examples/federatedml-1.x-examples) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080。

请忽略以下提示：

1. WARNING: Image for service mysql was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.

2. debconf: delaying package configuration, since apt-utils is not installed.

3. WARNING: You are using pip version 19.2.1, however version 19.2.2 is available.You should consider upgrading via the 'pip install --upgrade pip' command.

4. WARNING: Image for service xxx was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.

    

