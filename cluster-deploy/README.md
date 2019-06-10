 

#                      **Federal Learning FATE-V0.2 Deployment Guide**





## 1.     Module Information

In a party, FATE (Federated AI Technology Enabler) has the following 8 modules, including 7 offline related modules and 1 online related module. The specific module information is as follows:

### **1.1. Offline Module**

| Module Name         | Port of module | Method of deployment                                | Module function                                              |
| ------------------- | -------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Federation          | 9394           | Single node deployment in one party                 | Federation module handles task data communication (i.e. 'federation') among Federated |
| Meta-Service        | 8590           | Single node deployment in one party                 | Meta-Service module stores metadata required by this arch.   |
| Proxy               | 9370           | Single node deployment in one party                 | Proxy (Exchange) is communication channel among parties.     |
| Roll                | 8011           | Single node deployment in one party                 | Roll module is responsible for accepting distributed job submission, job / data schedule and result aggregations. |
| Storage-Service-cxx | 7778           | Party Multi-node deployment                         | Storage-Service module handles data storage on that single node. |
| Egg-Processor       | 7888           | Party Multi-node deployment                         | Processor is used to execute user-defined functions.         |
| Task-Manager        | 9360/9380      | Single node deployment in one party current version | Task Manager is a service for managing tasks. It can be used to start training tasks, upload and download data, publish models to serving, etc. |

### **1.2. Online Module**

| Module Name    | Port of module | Method of deployment                            | Module function                                              |
| -------------- | -------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Serving-server | 8001           | Party Multi-node deployment (two or more nodes) | Serving-Server is a online service for serving federated learning models. |



##2.     Deployment Architecture

### **2.1. Unilateral Deployment Architecture (for offline modules only)**

```
Meta-Service==============
`                        ||
Egg1============         ||=====Federation=====Proxy=====Firewall=====Other Parties or Exchange
`             ||         ||
Eggk======== Roll ========
`             ||
Eggn==========

			              Example deployment in one party
```

### **2.2. Module Deployment Method**

FATE is a network-based multi-node distributed deployment complex architecture. Because the online module configuration depends on the offline module in the current version, it can be deployed both online and offline through deployment scripts during actual deployment, only in offline deployment. Configure the serving-server role ip in the configurations.sh configuration file. In the deployment process, we can divide into two deployment structures according to whether the exchange role is included in the project:

A) Exchange role exists: each party communicates as a proxy through exchange.

B) There is no exchange role: the party directly connects to the other party's proxy role for communication.

*<u>Note: Exchanged must be available in the configuration.sh configuration file during unilateral deployment. If the exchange role does not exist, provide the server ip of the party with which you want to communicate as the exchange port;</u>*

Each party can be deployed on one or more servers. In reality, you can balance the flexible deployment of modules that can be deployed on a single node based on the number of servers provided. In order to facilitate the use in actual applications, this guide describes the two exchange roles of unilateral deployment and bilateral deployment.



## 3.     Installation Preparation

### **3.1. Server Configuration**

The following configuration is a one-sided server configuration information. If there are multiple parties, please refer to this configuration to copy this environment:

| Server                 |                                                              |
| ---------------------- | ------------------------------------------------------------ |
| **Quantity**           | 1 or more than 1 (according to the actual server allocation module provided) |
| **Configuration**      | 16 core / 32G memory / 300G hard disk / 50M bandwidth        |
| **Operating System**   | Version: CentOS Linux release 7.2                            |
| **Dependency Package** | yum source gcc gcc-c++ make autoconfig openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-dev (They can be installed using the initialization script env.sh) |
| **Users**              | User: app owner:apps (app user can sudo su root without password) |
| **File System**        | 1. The 300G hard disk is mounted to the /data directory.                                                                                2. Created /data/projects directory, projects directory belongs to app:apps |

### **3.2. Software Version Requirements**

There are different software environment requirements for the operation node and the node where the different modules are located. The following are divided into three types according to the software environment: 

a). Execution node: that is, we want to enter the node that executes the command;

b). Meta-Service: That is, the node where the Meta-Service role is located; 

c).Other Modules: The node where other modules are installed.

*<u>Note: The above nodes can be the same node, but to distinguish the description, the following is explained in the case of full distribution.</u>*

| node           | Node description                                  | Software configuration                                       | Software installation path                                   | Network Configuration                                        |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Execution node | The operation node that executes the script       | Git tool   Maven3.5 and above                                | Install it using the yum install command.                    | Interworking with the public network, you can log in to other node app users without encryption. |
| Meta-Service   | The node where the meta service module is located | Jdk1.8+       Python3.6  python virtualenv mysql8.0      redis5.0.2 | /data/projects/common/jdk/jdk1.8  /data/projects/common/miniconda3  /data/projects/fate/venv  /data/projects/common/mysql/mysql-8.0 /data/projects/common/redis/redis-5.0.2 | In the same area or under the same VPC as other nodes        |
| Other Modules  | Node where other modules are located              | Jdk1.8+  Python3.6  python virtualenv redis5.0.2             | /data/projects/common/jdk/jdk1.8 /data/projects/common/miniconda3 /data/projects/fate/venv /data/projects/common/redis/redis-5.0.2 | In the same area or under the same VPC as other nodes        |

Check whether the above software environment is reasonable in the corresponding server. If the software environment already exists and the correct installation path corresponds to the above list, you can skip this step. If not, refer to the following initialization steps to initialize the environment:

### **3.3. Package Preparation**

List of software involved:mysql-8.0、jdk1.8、python virtualenv surroundings  、Python3.6 Version: In order to facilitate the installation, a simple installation script is provided in the project. You can find the relevant script in the FATE/cluster-deploy/scripts/ fate-base directory of the project, download the corresponding version of the package and put it in the corresponding directory. The script is used to install the corresponding software package (because the software package is too large, you need to download it yourself during the actual installation process, so only the script file and txt file are kept here)
The script directory structure is as follows:

```
fate-base
|-- conf
|   `-- my.cnf
|-- env.sh
|-- install_java.sh
|-- install_redis.sh
|-- install_mysql.sh
|-- install_py3.sh
|-- packages
|   |-- jdk-8u172-linux-x64.tar.gz
|   |-- redis-5.0.2.tar.gz
|   |-- Miniconda3-4.5.4-Linux-x86_64.sh
|   `-- mysql-8.0.13-linux-glibc2.12-x86_64.tar.xz
|-- pip-dependencies
|   `-- requirements.txt
`-- pips
|   |-- pip-18.1-py2.py3-none-any.whl
|   |-- setuptools-40.6.3-py2.py3-none-any.whl
|   |-- virtualenv-16.1.0-py2.py3-none-any.whl
|  `-- wheel-0.32.3-py2.py3-none-any.whl
```
According to the above directory structure, the software packages and files needed by Python are placed in the corresponding directory. The dependency packages required by Python are given in the requirements. TXT file as a list. After downloading the dependency packages list, it can be placed in the pip-dependencies directory side by side with requirements. txt. The requirements.txt file can be obtained from [FATE/requirements.txt](https://github.com/WeBankFinTech/FATE/blob/master/requirements.txt) .

### **3.4. Software environment initialization**

After uploading, you can put the above-mentioned fate-base directory with dependencies into a fate-base .tar package and put it in the /data/app (optional) directory of the target server, and then decompress it:

```
cd /data/app
tar –xf fate-base.tar
cd fate 
```

If there is no app user, you need to create an app user:

```
groupadd -g 6000 apps
useradd -s /bin/sh -g apps -d /home/app app
passwd app
```

After the creation is completed, modify the app user password (according to actual needs)

Initialize the server and execute it under **root user**:

```
sh env.sh
```

The following steps are performed under the **app user**.

Check if jdk1.8 is installed. If the install_java.sh script is not installed, install it:

```
sh install_java.sh 
```

If you need to install the online module, check whether redis 5.0.2 is installed or not, and if not, execute the install_redis.sh script to install it:

```
sh install_redis.sh
```

*<u>Note:Using this script installation will initialize redis password to fate1234, which can be configured manually according to the actual situation.</u>*

Check Python 3.6 and the virtualized environment. If it is not installed, execute the install_py3.sh script to install it:

```
sh install_py3.sh
```

Check if mysql8.0 is installed. If it is not installed, execute the install_mysql.sh script to install it:

```
sh install_mysql.sh 
```

After installing mysql, **change the mysql password to "Fate123#$" and create database user "fate"** (modified according to actual needs):

```
$/data/projects/common/mysql/mysql-8.0.13/bin/mysql -uroot -p -S /data/projects/common/mysql/mysql-8.0.13/mysql.sock
Enter password:(please input the original password)
>set password='Fate123#$';
>CREATE USER 'fate'@'localhost' IDENTIFIED BY 'Fate123#$';
>GRANT ALL ON *.* TO 'fate'@'localhost';
>flush privileges;
```

After installing mysql, you need to use the following statement on the node where MySQL is installed to empower all IP in the party (replacing IP with actual ip):

```
$/data/projects/common/mysql/mysql-8.0.13/bin/mysql -ufate -p –S /data/projects/common/mysql/mysql-8.0.13/mysql.sock
Enter password: Fate123#$
>CREATE USER 'fate'@'$ip' IDENTIFIED BY 'Fate123#$';
>GRANT ALL ON *.* TO 'fate'@'$ip';
>flush privileges;
```

After the check is completed, return to the execution node for project deployment.



## 4.      Project Deployment

*<u>Note: The default installation directory is /data/projects/, and the execution user is app. It is modified according to the actual situation during installation.</u>*

### **4.1. Project Pull**

Go to the /data/projects/ directory of the execution node and execute the git command to pull the project from github:

```
cd /data/projects/
git clone https://github.com/WeBankFinTech/FATE.git
```

### **4.2. Maven Packaging**

Go into the project's arch directory and do dependency packaging:

```
cd FATE/arch
mvn clean package -DskipTests
cd FATE/fate-serving
mvn clean package -DskipTests
```

```
cd FATE/
git submodule update --init --recursive
```

*<u>Note:This step "git submodule update --init --recursive" takes a long time</u>*

You can also instead of "git submodule update --init --recursive"  by  downloade the list of  [glog-0.4.0](https://github.com/google/glog/tree/96a2f23dca4cc7180821ca5f32e526314395d26a)

、[grpc-1.19.1](https://github.com/grpc/grpc/tree/109c570727c3089fef655edcdd0dd02cc5958010)、[boost-1.68.0](https://github.com/boostorg/boost/tree/8f9a1cf1d15d262e09c16a305034d8bc1e39aca2)、[lmdb-0.9.22](https://github.com/LMDB/lmdb/tree/2a5eaad6919ce6941dec4f0d5cce370707a00ba7)、[protobuf-3.6.1](https://github.com/protocolbuffers/protobuf/tree/ca21b28287871660057a2b8bada2c044b6b3075d) yourself if conditions permit.

nd put them in the corresponding directory as the following directory tree: FATE/arch/eggroll/storage-service-cxx/third_party:

```
third_party
|--- boost
|--- glog
|--- grpc
|--- lmdb
|   `- libraries
|   |  `- liblmdb 
|--- protobuf
```



### **4.3. Modify Configuration File**

Go to the FATE/cluster-deploy/scripts directory in the FATE directory and modify the configuration file configurations.sh.
The configuration file configurations.sh instructions:

| Configuration item | Configuration item meaning                           | Configuration Item Value                                     | Explanation                                                  |
| ------------------ | ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| user               | Server operation username                            | Default is app                                               | Use the default value                                        |
| dir                | Fate installation path                               | Default is  /data/projects/fate                              | Use the default value                                        |
| mysqldir           | Mysql installation directory                         | Default is  /data/projects/common/mysql/mysql-8.0            | Mysql installation path can use the default value            |
| javadir            | JAVA_HOME                                            | Default is  /data/projects/common/jdk/jdk1.8                 | Jdk installation path can use the default value              |
| venvdir            | Python virtualenv installation directory             | Default is /data/projects/fate/venv                          | Use the default value                                        |
| redispass          | The password of redis                                | Default is fate1234                                          | Use the default value                                        |
| partylist          | Party.id array                                       | Each array element represents a partyid                      | Modify according to partyid                                  |
| JDBC0              | Corresponding to the jdbc configuration of the party | The corresponding jdbc configuration for each party: from left to right is ip dbname username password (this user needs to have create database permission) | If there are multiple parties, the order is JDBC0, JDBC1...the order corresponds to the partid order. |
| iplist             | A servers list of each party                         | Represents a list of server IPS contained in each party (except exchange roles) | All parties involved in the IP are placed in this list, repeat the IP once.All parties involved in the IP are placed in this list, repeat the IP once. |
| fedlist0           | Federation role IP list                              | Represents a list of servers with Federation roles in the party (only one in the current version) | If there are more than one party, the order is fedlist0, fedlist1... Sequence corresponds to partyid |
| meta0              | Meta-Service role  IP list                           | Represents a list of servers with Meta-Service roles in the party (only one in the current version) | If there are more than one party, the order is meta0, meta1... Sequence corresponds to partyid |
| proxy0             | Proxy role IP list                                   | Represents a list of servers with Proxy roles in the party (only one in the current version) | If there are more than one party, the order is proxy0, proxy1... Sequence corresponds to partyid |
| roll0              | Roll role IP list                                    | Represents a list of servers with Roll roles in the part (only one in the current version) | If there are more than one party, the order is roll0, roll1... Sequence corresponds to partyid |
| egglist0           | Egg role list                                        | Represents a list of servers included in each party          | If there are multiple parties, the order is egeglist0, the order of egglist1... corresponds to the order of partyid |
| tmlist0            | The ip list of the Task-manager role                 | Represents a list of servers with Roll roles in the party (only one in the current version) | If there are more than one party, the order is tmlist0, tmlist1... Sequence corresponds to partyid |
| serving0           | Serving-server role ip list                          | Each party contains a list of Serving-server roles ip        | If there are multiple parties, the order is serving0, serving1...corresponding to the partyid |
| exchangeip         | Exchange role ip                                     | Exchange role ip                                             | If the exchange role does not exist in the bilateral deployment, it can be empty. At this time, the two parties are directly connected. When the unilateral deployment is performed, the exchange value can be the proxy or exchange role of the other party. |

*<u>Note: tmipList and serving0, serving1 need to be configured only when online deployment is required, and configuration is not required only for offline deployment.</u>*

### **4.4. For Example**

Assume that each configuration is represented by the following code relationship:

| Code representation | Code node description                                        |
| ------------------- | ------------------------------------------------------------ |
| partyA.id           | Indicates the partyid of partyA                              |
| A.MS-ip             | Indicates the node where the meta-Service module of partyA is located. |
| A.F-ip              | Indicates the ip of the node where the federation module of partyA is located. |
| A.P-ip              | Indicates the ip of the node where partyA's Proxy module is located. |
| A.R-ip              | Indicates the ip of the node where the party module's Roll module is located. |
| A.TM-ip             | Indicates the server ipexchange where the partyA's Task-manager is located. |
| A.S-ip              | Indicates the server ip of the partyA's Serving-server. If there are multiple, the number is incremented. |
| A.E-ip              | Indicates the server ip of partyA's Egg. If there are more than one, add the number increment. |
| exchangeip          | Exchange server ip                                           |

The role code representation in partyB is similar to the above description.

*<u>Note: The above ip is based on the actual server ip assigned by each module. When the module is assigned to the same server, the ip is the same. Since the Egg-Storage-Service module and the Egg-Processor module are deployed at all nodes in the party, no special instructions are given.</u>*

According to the above table, the configuration.sh configuration file can be modified like this:

```
user=app 
dir=/data/projects/fate 
mysqldir=/data/projects/common/mysql/mysql-8.0 
javadir=/data/projects/common/jdk/jdk1.8 
venvdir=/data/projects/eggroll/venv
redispass=fate1234
partylist=(partyA.id partyB.id) 
JDBC0=(A.MS-ip A.dbname A.user A.password) 
JDBC1=(B.MS-ip B.dbname B.user B.password) 
iplist=(A.F-ip A.MS-ip A.P-ip A.R-ip B.F-ip B.MS-ip B.P-ip B.R-ip) 
fedlist0=(A.F-ip)
fedlist1=(B.F-ip)
meta0=(A.MS-ip)
meta1=(B.MS-ip)
proxy0=(A.P-ip)
proxy1=(B.P-ip)
roll0=(A.R-ip)
roll1=(B.R-ip)
egglist0=(A.E1-ip A.E2-ip A.E3-ip...)
egglist1=(B.E1-ip B.E2-ip B.E3-ip...) 
tmlist0=(A.TM-ip)
tmlist1=(B.TM-ip)
serving0=(A.S1-ip A.S2-ip)
serving1=(B.S1-ip B.S2-ip)
exchangeip=exchangeip 
```

*<u>Note: According to the above configuration method, you can modify it according to the actual situation.</u>*

After modifying the configuration items corresponding to the configurations.sh file according to the above configuration, execute the auto-packaging.sh script:

```
cd /data/projects/FATE/cluster-deploy/scripts
bash auto-packaging.sh
```

This script file puts each module and configuration file into the FATE/cluster-deploy/example-dir-tree directory. You can view the directory and files of each module in this directory as following example-dir-tree:

```
example-dir-tree
|
|--- federation
|    |- conf/
|    |  |- applicationContext-federation.xml
|    |  |- federation.properties
|    |  |- log4j2.properties
|    |
|    |- lib/
|    |- fate-federation-0.2.jar
|    |- fate-federation.jar -> fate-fedaration-0.2.jar
|    |- service.sh
|
|--- meta-service
|    |- conf/
|    |  |- applicationContext-meta-service.xml
|    |  |- jdbc.properties
|    |  |- log4j2.properties
|    |  |- meta-service.properties
|    |
|    |- lib/
|    |- fate-meta-service-0.2.jar
|    |- fate-mata-service.jar -> fate-meta-service-0.2.jar
|    |- service.sh
|
|--- proxy
|    |- conf/
|    |  |- applicationContext-proxy.xml
|    |  |- log4j2.properties
|    |  |- proxy.properties
|    |  |- route_table.json
|    |
|    |- lib/
|    |- fate-proxy-0.2.jar
|    |- fate-proxy.jar -> fate-proxy-0.2.jar
|    |- service.sh
|
|--- python
|    |- arch
|    |  |- api/
|    |  |- conf/
|    |  |- processor/
|    |  |- task_manager/
|    |  |- eggroll/
|    |
|    |- federatedml/
|    |- examples/
|    |- workflow/
|    |- processor.sh
|    |- service.sh
|
|--- roll
|    |- conf/
|    |  |- applicationContext-roll.xml
|    |  |- log4j2.properties
|    |  |- roll.properties
|    |
|    |- lib/
|    |- fate-roll-0.2.jar
|    |- fate-roll.jar -> fate-roll-0.2.jar
|    |- service.sh
|
|--- storage-service
|    |- conf/
|    |  |- log4j2.properties
|    |
|    |- lib/
|    |- fate-storage-service-0.2.jar
|    |- fate-storage-service.jar -> fate-storage-service-0.2.jar
|    |- service.sh
|
|--- storage-service-cxx
|    |- logs/
|    |- Makefile
|    |- proto
|    |  |- storage.proto
|    |
|    |- src/
|    |- third_party
|    |  |- boost/
|    |  |- glog/
|    |  |- grpc/
|    |  |- lmdb/
|    |  |- protobuf/
|    |
|    |- storage-service.cc
|
|--- serving-server
|    |- conf/
|    |  |- log4j2.properties
|    |  |- serving-server.properties
|    |
|    |- lib/
|    |- fate-serving-server-0.2.jar
|    |- fate-serving-server.jar -> fate-serving-server-0.2.jar
|    |- service.sh
```

Continue to execute the deployment script in the FATE/cluster-deploy/scripts directory:

```
cd /data/projects/FATE/cluster-deploy/scripts
bash auto-deploy.sh
```



## 5.     Configuration Check

After the execution, you can check whether the configuration of the corresponding module is accurate on each target server. Users can find a detailed configuration document in [cluster-deploy/doc](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/doc) .



## 6.     Start And Stop Service

### 6.1. Startup service

Use ssh to log in to each node with **app user**. Go to the /data/projects/fate directory and run the following command to start all services:

```
cd  /data/projects/fate
sh services.sh all start
```

If the server is a serving-server node, you also need:

```
cd /data/projects/fate/serving-server
sh service.sh start
```

If the server is a task_manager node, you also need:

```
cd /data/projects/fate/python/arch/task_manager
sh service.sh start
```

*<u>Note: If the target environment cannot install the c++ environment, you can replace storage-serivice-cxx in the services.sh file with storage-serivice and restart it. You can use the Java version of storage-service module.</u>*

### 6.2. Check service status

Check whether each service process starts successfully:

```
cd  /data/projects/fate
sh services.sh all status
```

If the server is a serving-server node, you also need:

```
cd /data/projects/fate/serving-server
sh service.sh status
```

If the server is a task_manager node, you also need:

```
cd /data/projects/fate/python/arch/task_manager
sh service.sh status
```

### 6.3. Shutdown service

To turn off the service, use:

```
cd  /data/projects/fate
sh services.sh all stop
```

If the server is a serving-server node, you also need:

```
cd /data/projects/fate/serving-server
sh service.sh stop
```

If the server is a task_manager node, you also need:

```
cd /data/projects/fate/python/arch/task_manager
sh service.sh stop
```

*<u>Note: If there is a start and stop operation for a single service, replace all in the above command with the corresponding module name.</u>*



## 7.      Test

### **7.1. Stand-alone Test**

Use ssh login to each node **app user**, enter the /data/projects/fate directory to execute:

```
source /data/projects/fate/venv/bin/activate
export PYTHONPATH=/data/projects/fate/python
cd $PYTHONPATH
sh ./federatedml/test/run_test.sh
```

See the "OK" field to indicate that the operation is successful.In other cases, if FAILED or stuck, it means failure, the program should produce results within one minute.

### **7.2. Toy_example Deployment Verification**

**Stand-alone version:** 

Start the virtualized environment Run run_toy_examples_standalone.sh under /data/projects/fate/python/examples/toy_examples:

```
export PYTHONPATH=/data/projects/fate/python 
source /data/projects/fate/venv/bin/activate
cd /data/projects/fate/python/examples/toy_example/
sh run_toy_example_standalone.sh
```

See the "OK" field to indicate that the operation is successful.In other cases, if FAILED or stuck, it means failure, the program should produce results within one minute.

**Distributed version:**
Start the virtualized environment in host and guest respectively. Run run_toy_example_cluster.sh under /data/projects/fate/python/examples/toy_examples.

```
export PYTHONPATH=/data/projects/fate/python 
source /data/projects/fate/venv/bin/activate
cd /data/projects/fate/python/examples/toy_example/
```

In the host party, running:

```
sh run_toy_example_cluster.sh host $jobid guest_parityid host_partyid
```

In the guest party, running: 

```
sh run_toy_example_cluster.sh guest $jobid guest_parityid host_partyid
```

See the "OK" field to indicate that the operation is successful.
In other cases, if FAILED or stuck, it means failure, the program should produce results within one minute.

### 7.3. Minimization testing

Start the virtualization environment in host and guest respectively and run run under / data / projects / fate / Python / examples / min_test_task:

```
export PYTHONPATH=/data/projects/fate/python
source/data/projects/fate/venv/bin/activate
cd/data/projects/fate/python/examples/min_test_task/
```

**Fast mode**

In the task_manager node of host part, running:

```
sh run.sh host fast 
```

In the task_manager node of guest part, running: 

```
sh run.sh guest fast 
```

Wait a few minutes, see the result show "success" field to indicate that the operation is successful.
In other cases, if FAILED or stuck, it means failure.

**Normal mode**

In the task_manager node of host part, running:

```
sh run.sh host normal 
```

In the task_manager node of guest part, running: 

```
sh run.sh guest normal 
```

Waiting for ten minutes, see the result show "success" field to indicate that the operation is successful.
In other cases, if FAILED or stuck, it means failure.