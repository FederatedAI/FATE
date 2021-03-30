## ****FATE Stand-alone Deployment Guide****

Server Configuration;

| **Quantity**           |    1                                                  |
| ---------------------- | ----------------------------------------------------- |
| **Configuration**      | 8 core / 16G memory / 500G hard disk                  |
| **Operating System**   | Version: CentOS Linux release 7                       |
| **Users**              | User: app owner:apps                                  |

The stand-alone version provides 2 deployment methods, which can be selected according to your actual situation:

- Install FATE using Docker  *(Recommended)* 

- Install FATE  in Host 

You can also refer to [Chinese guide](./doc/Fate-standalone_deployment_guide_zh.md) 


#### 1) Install FATE using Docker*(Recommended)* 

It is strongly recommended to use docker, which greatly reduces the possibility of encountering problems.

1. The host needs to be able to access the external network,pull the installation package and docker image from the public network.

2. Dependent on [docker](https://download.docker.com/linux/) , docker recommended version is 18.09, you can use the following command to verify the docker environment: docker --version , docker start and stop and other Please refer to: docker --help.

3. Keep the 8080 port accessible before executing. If you want to execute again, please delete the previous container and image with the docker command.

   please follow the below step:


```
#Please replace ${version} below with the real version you want to use!

#Get code
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone_fate_${version}.tar.gz
tar -xzvf docker_standalone_fate_${version}.tar.gz

#Execute the command
cd docker_standalone_fate_${version}
bash install_standalone_docker.sh
```

4. Test

   - Unit Test

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate"`
   docker exec -t -i ${CONTAINER_ID} bash
   bash ./python/federatedml/test/run_test.sh
   ```

   If success,  the screen shows like blow:

   ```
   there are 0 failed test
   ```

   - Toy_example Test

   ```
   CONTAINER_ID=`docker ps -aqf "name=fate"`
   docker exec -t -i ${CONTAINER_ID} bash
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   If success,  the screen shows like blow:

   ```
   success to calculate secure_sum, it is 2000.0
   ```

5. Install FATE-Client and FATE-Test

   To conveniently interact with FATE, we provide tools [FATE-Client](../python/fate_client) and [FATE-Test](../python/fate_test).

   Install FATE-Client and FATE-Test with the following commands:

   ```
    pip install fate-client
    pip install fate-test
   ```
   

There are a few algorithms under [examples](../examples/dsl/v2) folder, try them out!

You can also experience the fateboard access via a browser:
Http://hostip:8080.



#### 2) Install FATE in Host

1. Check whether the local 8080,9360,9380 port is occupied.

   ```
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

2. Download the compressed package of stand-alone version and decompress it.

   ```
   #Please replace ${version} below with the real version you want to use!
   
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/standalone_fate_master_${version}.tar.gz
   tar -xzvf  standalone_fate_master_${version}.tar.gz
   ```

3. Enter FATE directory and execute the init.sh.

   ```
   #Please replace ${version} below with the real version you want to use!
   
   cd standalone_fate_master_${version}
   sh init.sh init
   ```

4. Test

   - Unit Test

   ```
   #Please replace ${version} below with the real version you want to use!
   
   cd standalone_fate_master_${version}
   source bin/init_env.sh
   bash ./python/federatedml/test/run_test.sh
   ```

   If success,  the screen shows like blow:

   ```
   there are 0 failed test
   ```

   - Toy_example Test

   ```
   #Please replace ${version} below with the real version you want to use!
   
   cd standalone_fate_master_${version}
   source bin/init_env.sh
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   If success,  the screen shows like blow:

   ```
   success to calculate secure_sum, it is 2000.0
   ```

5. Install FATE-Client and FATE-Test

   To conveniently interact with FATE, we provide tools [FATE-Client](../python/fate_client) and [FATE-Test](../python/fate_test).

   Install FATE-Client and FATE-Test with the following commands:

   ```
   python -m pip install fate-client
   python -m pip install fate-test
   ```


There are a few algorithms under [examples](../examples/dsl/v2) folder, try them out!

You can also experience the fateboard access via a browser:
Http://hostip:8080.

