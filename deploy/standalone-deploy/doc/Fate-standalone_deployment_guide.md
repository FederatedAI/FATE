# FATE单机部署指南(仅仅为占位，需要翻译)

[TOC]

## 1. 版本历史

| 版本状态 | 创建人     |   完成日期 | 备注 |
| :------- | :--------- | ---------: | :--- |
| 1.0      | jarviszeng | 2021-11-10 | 初始 |

## 2. 说明

服务器配置：

| **数量**      |    1                                                  |
| ------------ | ----------------------------------------------------- |
| **配置**      | 8 core / 16G memory / 500G hard disk                  |
| **操作系统**   | Version: CentOS Linux release 7                       |
| **用户**      | User: app owner:apps                                  |

单机版提供2种部署方式，可以根据实际情况选择：

- 使用Docker镜像安装FATE（推荐）

- 在主机中安装FATE

## 3. 使用Docker镜像安装FATE（推荐）

建议使用docker镜像，这样可以大大降低遇到问题的可能性

1. 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。

2. 依赖[docker](https://download.docker.com/linux/)，docker建议版本为18.09，您可以使用以下命令验证docker环境：docker --version,docker的起停和其他操作请参考docker --help。

3. 执行之前，请检查8080是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

   请按照以下步骤操作:

    注意，请用实际的版本号替换下文中的${version},参考[fate.env](../../../fate.env)文件中的FATE版本！

   ```   
   #获取安装包
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone_fate_${version}.tar.gz
   tar -xzvf docker_standalone_fate_${version}.tar.gz
   
   #执行部署
   cd docker_standalone_fate_${version}
   bash install_standalone_docker.sh
   ```

4. 测试

   - 单元测试

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate"`
   docker exec -t -i ${CONTAINER_ID} bash
   bash ./python/federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   there are 0 failed test
   ```

   - Toy测试

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate"`
   docker exec -t -i ${CONTAINER_ID} bash
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   success to calculate secure_sum, it is 2000.0
   ```

5. 安装FATE-Client和FATE-Test
   
   为方便使用FATE，我们提供了便捷的交互工具[FATE-Client](../../../python/fate_client)以及测试工具[FATE-Test](../../../python/fate_test).
   
   请在环境内使用以下指令安装：
   
   ```
    pip install fate-client
    pip install fate-test
   ```


有些用例算法在 [examples](../../../examples/dsl/v2) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080。

## 4. 在主机中安装FATE

注意，如下示例中的${version}，请用实际的版本号替换，参考[fate.env](../../../fate.env)文件中的FATE版本！

### 4.1 检查

本地8080、9360、9380端口是否被占用

   ```bash
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

### 4.2 获取安装包

下载安装包并解压缩

   ```bash
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/1.7.0/release/standalone_fate_install_${version}_release.tar.gz
   tar -xzvf standalone_fate_install_${version}_release.tar.gz
   ```

### 4.3 自动初始化

进入解压后的目录并使用init.sh进行初始化

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
   cd standalone_fate_install_${version}_release
   sh init.sh init
   ```

### 4.4 启动

   ```bash
   sh init.sh status
   sh init.sh start
   ```

### 4.5 测试

   - 加载环境变量

   ```bash
   source bin/init_env.sh
   ```

   - Toy测试

   ```bash
   flow test toy --guest-party-id 10000 --host-party-id 10000
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   success to calculate secure_sum, it is 2000.0
   ```

   - 单元测试

   ```bash
   bash ./fate/python/federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   there are 0 failed test
   ```
