## ****FATE Stand-alone Deployment Guide****

The stand-alone version provides three deployment methods, which can be selected according to your actual situation:

- Install FATE using Docker [Chinese guide](./doc/Fate-V1.1单机版部署指南.md) *(Recommended)* 

- Install FATE  in Host [Chinese guide](./doc/Fate-V1.1单机版部署指南.md) 

- Build FATE from Source with Docker [Chinese guide](./doc/Fate-V1.1单机版部署指南.md)(It takes 40 minutes or more)

  

#### 1) Install FATE using Docker*(Recommended)* 

It is strongly recommended to use docker, which greatly reduces the possibility of encountering problems.

1. The host needs to be able to access the external network,pull the installation package and docker image from the public network.

2. Dependent on docker and docker-compose, You can verify the docker environment with the following command:docker --version and docker-compose --version.

3. execute follow command by root user (because need to create directories like /var/lib/fate/data).

4. Check whether the 8080, 9060, and 9080 ports are occupied before executing. If you want to execute again, please delete the previous container and image with the docker command.

   please follow the below step:


```
#Get code
FATE $ wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone.tar.gz
FATE $tar -xvf docker_standalone.tar.gz

#Execute the command
FATE $ cd docker_standalone
FATE $ bash install_standalone_docker.sh

#Validation results
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh

```

There are a few algorithms under [examples](https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples) folder, try them out!

You can also experience the fateboard access via a browser:
Http://hostip:8080.



#### 2) Install FATE  in Host

1. Check whether the local 8080,9360,9380 port is occupied.

   ```
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

2. Download the compressed package of stand-alone version and decompress it. 

   ```
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE.tar.gz
   tar -xvf  FATE.tar.gz
   ```

3. Enter FATE directory and execute the init.sh.

   ```
   cd FATE
   source init.sh
   ```

4. Execution test.

   ```
   cd FATE
   bash ./federatedml/test/run_test.sh
   ```

There are a few algorithms under [examples](https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples) folder, try them out!

You can also experience the fateboard access via a browser:
Http://hostip:8080.



#### 3) Build FATE from Source with Docker

1. The host needs to be able to access the external network,pull the installation package and docker image from the public network.

2. Dependent on docker and docker-compose, You can verify the docker environment with the following command:docker --version and docker-compose --version.

3. execute follow command by root user (because need to create directories like /var/lib/fate/data).

4. Check whether the 8080, 9060, and 9080 ports are occupied before executing. If you want to execute again, please delete the previous container and image with the docker command.

5. It takes about 40 minutes to complete the execution, please wait for a moment.

   please follow the below step:

```
#Get code
FATE $ git clone https://github.com/WeBankFinTech/FATE.git

#Execute the command
FATE $ cd FATE/standalone-deploy
FATE $ bash build_standalone_docker.sh

#Validation results
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh

```

There are a few algorithms under [examples](https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples) folder, try them out!

You can also experience the fateboard access via a browser:
Http://hostip:8080.

Please ignore the following tips:

1. debconf: delaying package configuration, since apt-utils is not installed.

2. WARNING: You are using pip version 19.2.1, however version 19.2.2 is available.You should consider upgrading via the 'pip install --upgrade pip' command.

3. WARNING: Image for service xxx was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.

     

