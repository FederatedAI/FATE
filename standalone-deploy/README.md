## ****FATE Stand-alone Deployment Guide****

####  1. Server Configuration

​	The following configuration is a one-sided server configuration information. If there are multiple parties, please refer to this configuration to copy this environment:

| Server               |                                                              |
| -------------------- | ------------------------------------------------------------ |
| **Quantity**         | 1 (according to the actual server allocation module provided) |
| **Configuration**    | 8 core / 16G memory / 500G hard disk / 10M bandwidth         |
| **Operating System** | Version: CentOS Linux release 7.2                            |
| **Users**            | User: app owner:apps (app user can sudo su root without password) |
| **File System**      | 1. The 500G hard disk is mounted to the /data directory. 2. Created /data/projects directory, projects directory belongs to app:apps. |

```
#Get code
FATE $ wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/docker_standalone-fate-1.3.0.tar.gz
FATE $tar -xvf docker_standalone-fate-1.3.0.tar.gz

#Execute the command
FATE $ cd docker_standalone-fate-1.3.0
FATE $ bash install_standalone_docker.sh

#Validation results
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ bash ./federatedml/test/run_test.sh

```

There are a few algorithms under [examples](../examples/federatedml-1.x-examples) folder, try them out!

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
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/standalone-fate-master-1.3.0.tar.gz
   tar -xvf  standalone-fate-master-1.3.0.tar.gz
   ```

3. Enter FATE directory and execute the init.sh.

   ```
   cd standalone-fate-master-1.3.0
   source init.sh init
   ```

4. Execution test.

   ```
   cd standalone-fate-master-1.3.0
   bash ./federatedml/test/run_test.sh
   ```

There are a few algorithms under [examples](https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples) folder, try them out!

You can also experience the fateboard access via a browser (hostip below represents your local ip):
Http://hostip:8080.