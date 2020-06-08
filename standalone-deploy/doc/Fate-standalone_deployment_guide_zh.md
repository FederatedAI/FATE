## ****FATE单机部署指南****

单机版提供三种部署方式，可以根据实际情况选择：

- 使用Docker镜像安装FATE（推荐）

- 在主机中安装FATE


#### 1) 使用Docker镜像安装FATE（推荐） 

建议使用docker镜像，这样可以大大降低遇到问题的可能性。

1. 主机需要能够访问外部网络，从公共网络中拉取安装包和docker镜像。

2. 依赖[docker](https://download.docker.com/linux/)和[docker-compose](https://github.com/docker/compose/releases/tag/1.24.0)，docker建议版本为18.09，docker-compose建议版本为1.24.0，您可以使用以下命令验证docker环境：docker --version和docker-compose --version，docker的起停和其他操作请参考docker --help。

3. 执行之前，请检查8080、9060和9080端口是否已被占用。 如果要再次执行，请使用docker命令删除以前的容器和镜像。

   请按照以下步骤操作:

   ```
   #获取安装包
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone-fate-1.4.1.tar.gz
   tar -xzvf docker_standalone-fate-1.4.1.tar.gz
   
   #执行部署
   cd docker_standalone-fate-1.4.1
   bash install_standalone_docker.sh
   ```

4. 测试

   - 单元测试

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate_python"`
   docker exec -t -i ${CONTAINER_ID} bash
   bash ./federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   there are 0 failed test
   ```

   - Toy测试

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate_python"`
   docker exec -t -i ${CONTAINER_ID} bash
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   success to calculate secure_sum, it is 2000.0
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
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/standalone-fate-master-1.4.1.tar.gz
   tar -xzvf  standalone-fate-master-1.4.1.tar.gz
   ```

3. 进入FATE目录并执行init.sh。

   ```
   cd standalone-fate-master-1.4.1
   source init.sh init
   ```

4. 测试

   - 单元测试

   ```
   cd standalone-fate-master-1.4.1
   bash ./federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   there are 0 failed test
   ```

   - Toy测试

   ```
   cd standalone-fate-master-1.4.1
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   如果成功，屏幕显示类似下方的语句:

   ```
   success to calculate secure_sum, it is 2000.0
   ```

   

有些用例算法在 [examples](../../examples/federatedml-1.x-examples) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080。
