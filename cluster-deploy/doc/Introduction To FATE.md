**Federal Learning FATE-V0.1 Deployment Detailed Design**

 

 

# 1.     Introduction To FATE

 

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

 

# 2.     Module Information

In one party，FATE（Federated AI Technology Enabler）has the following six modules, the module information is as follows:

| Module Name         | Port of module | Method of deployment                | Module function                                              |
| ------------------- | -------------- | ----------------------------------- | ------------------------------------------------------------ |
| Federation          | 9394           | Single node deployment in one party | Federation module handles task data communication   (i.e. 'federation') among Federated |
| Meta-Service        | 8590           | Single node deployment in one party | Meta-Service module stores metadata required by this   arch. |
| Proxy/Exchange      | 9370           | Single node deployment in one party | Proxy (Exchange) is communication channel among   parties.   |
| Roll                | 8011           | Single node deployment in one party | Roll module is responsible for accepting distributed   job submission, job / data schedule and result aggregations. |
| Egg-Storage-Service | 7778           | All nodes deployment in the party   | Storage-Service module handles data storage on that   single node. |
| Egg-Processor       | 7888           | All nodes deployment in the party   | Processor is used to execute user-defined functions.         |

 

# 3.     Architecture Of One Party

 

l  Deployment architecture in on party

 

Meta Service————

​                                       |

Egg1————Federation——Proxy———Firewall—Other Parties or Exchange

​            	|              |

Eggk——Roll ———

​                 |

Eggn——

Example deployment in one party

 

l  Method of module deployment

FATE is a complex network multi-node distributed deployment architecture. During the deployment process, we can divide it into two deployment modes according to whether exchange is included in the project or not.

A) Exchange role exists: each part communicates through exchange as a proxy;

B) There is no exchange role: Parts communicate directly with each other's proxy role.

Note: Exchange IP must be provided in the configuration file when unilateral deployment occurs. If the exchange role does not exist, the server IP where proxy is located in the part that needs to communicate with it should be provided as exchange ip.

Each part can be deployed on one or more servers. In practice, you can allocate flexible deployment of modules that can be deployed on a single node according to the number of servers provided. In order to facilitate the practical application, this guide describes unilateral deployment and bilateral deployment of exchange roles respectively.

# 4.     Installation Preparation

 

l  Server Configuration

The following configuration is the server configuration information of a single party. If there are multiple parties, copy this environment with reference to this configuration:

| **Server**                           |                                                              |
| ------------------------------------ | ------------------------------------------------------------ |
| **Number of servers**                | >1(Server allocation module based on actual provision)       |
| **Configuration**                    | 16 core / 32G memory / 300G hard disk / 50M bandwidth        |
| **Operating System and file system** | Dependency package: yum source gcc gcc-c++ make autoconfig openssl-devel   supervisor gmp-devel mpfr-devel libmpc-devel   Business users:   User: app Owner: apps   You can use the following commands to create:   groupadd -g 6000 apps   `useradd -s /bin/sh -g apps –d /home/app app`   passwd app    File system:   1. 300G hard disk mount to /data directory.   2. Created /data/projects directory, projects directory belongs to app:apps |

 

l  Requirements Of Software Version 

There are different software environment requirements for the operation node and the node where the different modules are located. The following are divided into three types according to the software environment: a). Execution node: that is, we want to enter the node that executes the command; b). Meta-Service: That is, the node where the Meta-Service role is located; c).Other Modules: The node where other modules are installed.

Note: The above nodes can be the same node, but to distinguish the description, the following is explained in the case of full distribution.

| Node           | Node Description                            | Software                                 | Software Installation Path                                   | Network Configuration                                        |
| -------------- | ------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Execution node | The operation node that executes the script | Git tool   Maven3.5 and above            | Install it with the yum install command.                     | Interworking with the public network, you can log in to other node app   users without encryption. |
| Meta-Service   | Meta-Service module node                    | Jdk1.8+    Python3.6    venv    mysql8.0 | /data/projects/common/jdk/jdk1.8.0_192    /data/projects/common/miniconda3    /data/projects/fate/venv    /data/projects/common/mysql/mysql-8.0.13 | In the same area or under the same VPC as other nodes   Communicate with other nodes   App user has secret sudo root permission |
| Other Modules  | Other modules node                          | Jdk1.8+    Python3.6    venv             | /data/projects/common/jdk/jdk1.8.0_192    /data/projects/common/miniconda3    /data/projects/fate/venv |                                                              |

 

l  Software Version Check

²  Check whether the above software environment is reasonable in the corresponding server. If the software environment already exists and the correct installation path corresponds to the above list, you can skip this step. If not, you can initialize the environment by uploading the software package and executing the corresponding script. , here are the initialization steps:

²  Software Packages Preparation

List of software involved:

fate-base.tar

 

²  Package Uploading

The package is generally uploaded to the /data/app directory and then decompressed:

cd /data/app

tar –xf fate-mysql.tar

cd fate 

Check whether the /data/projects/ fate directory exists. The app user operates in the operation of the fate directory. If not, execute the env.sh script:

sh env.sh

Check if jdk1.8 is installed. If it is not installed, execute the install_java.sh script to install it:

sh install_java.sh

Check Python 3.6 and the virtualized environment. If it is not installed, execute the install_py3.sh script to install it:

sh install_py3.sh

Check if mysql8.0 is installed. If it is not installed, execute the install_mysql.sh script to install it:

sh install_mysql.sh

After installing mysql, change the mysql password to Fate123#$ (modified according to actual needs).

After the check is completed, return to the execution node for project deployment.

# 5.     Project Deployment

Note: The default installation directory for this guidance is / data / projects /, and the execution user is app. The installation directory is modified according to the actual situation.

l  Pull Project

Go to the /data/projects/ directory of the execution node and execute the git command to pull the project from github:

cd /data/projects/

git clone <https://github.com/WeBankFinTech/FATE.git>

l  Maven Clean

Go into the project's arch directory and do dependency packaging:

cd FATE/arch

mvn clean package -DskipTests

 

l  Modify the Configuration File

Go to the FATE/cluster-deploy/scripts directory in the FATE directory and modify the configuration file configuration.sh.

The configuration file configurations.sh instructions:

| Configuration item | Configuration item meaning                             | Configuration Item Value                                     | Explanation                                                  |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| user               | server operation username                              | Defaults to app                                              | Use the default value                                        |
| dir                | Installation path of fate                              | Defaults to /data/projects/fate                              | Use the default value                                        |
| mysqldir           | Installation path of mysql                             | Defaults to   /data/projects/common/mysql/mysql-8.0.13       | Use the default value                                        |
| javadir            | JAVA_HOME path                                         | Defaults to    /data/projects/common/jdk/jdk1.8.0_192        | Use the default value                                        |
| partylist          | party.id Array                                         | Each array element represents a   partyid                    | Modify according to partyid                                  |
| JDBC0              | Corresponding to the jdbc   configuration of the party | The jdbc configuration for each party:   from left to right is ip dbname username password | If there are multiple parties, then   JDBC0, JDBC1... Corresponding to partid order   Fill according to jdbc configuration |
| roleiplist         | List of module servers                                 | The ip of the federation,   meta-service, proxy, and roll modules is shown from left to right. | If there are multiple parties, they   are placed behind the array. |
| egglist0           | List of egg module servers                             | Represents a list of egg-only role   servers installed       | If there are more than one part, the   order is egglist0, egglist1... Corresponding to partid order |
| exchangeip         | Exchange role ip                                       | Exchange role ip                                             | If there is no exchange role in   bilateral deployment, it can be empty, then both sides are directly   connected; when unilateral deployment, exchange value can be provided for   each other's proxy or exchange role, which must be provided. |

 

l  For Example

| Node Name | Node Description                 |
| --------- | -------------------------------- |
| MS-ip     | Meta-Service module node         |
| F-ip      | Ip of the Federation module node |
| P-ip      | Ip of the Proxy module node      |
| R-ip      | Ip of the Roll module node       |
| E-ip      | Ip of the Exchange module node   |

 

Note: The above ip is based on the actual server ip assigned by each module. When the module is assigned to the same server, the ip is the same. Since the Egg-Storage-Service module and the Egg-Processor module are deployed at all nodes in the party, no special instructions are given.

 

user=app (The server user, use default value)

dir=/data/projects/fate(The absolute path of the fate directory, use default value)

mysqldir=/data/projects/common/mysql/mysql-8.0.13(Mysql install absolute path, use default value)

javadir=/data/projects/common/jdk/jdk1.8.0_192(The absolute path of the java installation, use default value)

partylist=( partyA.id partyB.id) (Representing partyA partyB)

JDBC0=(A.ip A.dbname A.user A.password) ( partyA jdbc configuration)

JDBC1=( B.ip B.dbname B.user B.password) ( partyB jdbc configuration)

roleiplist=( F-ip MS-ip P-ip R-ip F-ip MS-ip P-ip R-ip) ( The module of partyA and partyB is located in ip)

egglist0=(e1.ip e2.ip …)( List of servers that only install egg roles in partyA)

egglist1=(e1.ip e2.ip …) ( List of servers that only install egg roles in partyB)

exchangeip=exchangeip(Exchange server ip,if exchange role does not exist: the default is empty for bilateral deployment; IP for server on the other side is filled in for unilateral deployment)

Note: Yellow is required to modify the configuration item

 

Modify the configuration items corresponding to the configurations.sh file according to the above configuration, and then execute the auto-packaging.sh script:

cd /data/projects/FATE/cluster-deploy/scripts

bash auto-packaging.sh

This script file puts each module and configuration file into the FATE/cluster-deploy/example-dir-tree directory, where you can view the directories and files of each module.

Continue to execute the deployment script in the FATE/cluster-deploy/scripts directory:

cd /data/projects/FATE/cluster-deploy/scripts

bash auto- deploy.sh

 

# 6.     Configuration Checking

After the execution, you can check whether the configuration of the corresponding module is accurate on each target server. The corresponding configuration files and configuration items of each module are as follows:

| Configuration   Files                          | Configuration   Item                                         | Configuration   Item Description                             |
| ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fate/federation/conf/federation.properties     | party.id                                                     | party   id of FL participant, e.g. 10000                     |
| service.port                                   | port of   federation module, defaults to 9394                |                                                              |
| meta.service.ip                                | ip of meta-service   module, e.g. 172.16.153.xx              |                                                              |
| meta.service.port                              | port of   meta-service, defaults to 8590                     |                                                              |
| fate/meta-service/conf/meta-service.properties | party.id                                                     | party   id of FL participant, e.g. 10000                     |
| service.port                                   | port of   meta-service module, defaults to 8590              |                                                              |
| fate/meta-service/conf/jdbc.properties         | jdbc.driver.classname                                        | jdbc   driver's classname, recommendation: com.mysql.cj.jdbc.Driver |
| jdbc.url                                       | jdbc   connection url, modify as needed                      |                                                              |
| jdbc.username                                  | database   username, modify as needed                        |                                                              |
| jdbc.password                                  | database   password, modify as needed                        |                                                              |
| target.project                                 | target   project. Required by mybatis-generator, fixed to meta-service |                                                              |
| fate/proxy/conf/proxy.properties               | coordinator                                                  | same as   party id, e.g. 10000                               |
| ip                                             | ip to   bind (in multi-interface env), optional              |                                                              |
| port                                           | port of   proxy (exchange) module, defaults to 9370          |                                                              |
| route.table                                    | path to   route table                                        |                                                              |
| fate/proxy/conf/route_table.json               | exchangeip   or proxy ip / 9370                              | "default":   {          "default": [            {              "ip": "ip of   proxy or exchange module",              "port": 9370            }          ]        }, |
| party.id   federaion ip/9394                   | "10000":   {          "fate": [            {              "ip": "ip of   federation module",              "port": 9394            }          ]        }, |                                                              |
| fate/roll/conf/roll.properties                 | party.id                                                     | party   id of FL participant, e.g. 10000                     |
| service.port                                   | port of   roll module, defaults to 8011                      |                                                              |
| meta.service.ip                                | ip of   meta-service module                                  |                                                              |
| meta.service.port                              | port of   meta-service module, defaults to 8590              |                                                              |
| fate/egg/conf/egg.properties                   | service.port                                                 | port of   egg module, defaults to 7888                       |
| processor.venv                                 | path of   venv, defaults to /data/projects/fate/venv         |                                                              |
| processor.path                                 | path of   processor, defaults to /data/projects/fate/python/arch/processor/processor.py |                                                              |
| python.path                                    | path of   python, defaults to /data/projects/fate/python     |                                                              |
| data.dir                                       | path of   data-dir, defaults to /data/projects/fate/data-dir |                                                              |
| fate/python/arch/conf/server_conf.json         | host of   Federation                                         | ip of   federation module                                    |
| port of   Federation                           | port of   federation module, defaults to 9394                |                                                              |
| host of   Roll                                 | ip of   roll module                                          |                                                              |
| port of   Roll                                 | port of   roll module, defaults to 8011                      |                                                              |

 

 

# 7.     Start And Stop Service

(A).Use ssh to log in to each node app user. Go to the /data/projects/fate directory and run the following command to start all services:

sh services.sh all start

(B).Check whether each service process starts successfully:

sh services.sh all status

(C).To turn off the service, use:

sh services.sh all stop

Note: If there is a start and stop operation for a single service, replace all in the above command with the corresponding module name.

# 8.     Test

Use ssh to log in to each node app user and go to the /data/projects/fate directory to execute:

source /data/projects/fate/venv/bin/activate

export PYTHONPATH=/data/projects/fate/python

cd $PYTHONPATH

sh ./federatedml/test/run_test.sh

python ./federatedml/test/eleven_demo.py

 

 

 