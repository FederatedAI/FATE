

#                      **FATE Deployment Guide**

The Cluster version provides four deployment methods, which can be selected according to your actual situation:

- Install Cluster Step By Step [Chinese guide](./doc/Fate_step_by_step_install_zh.md) 
- Install AllinOne [Chinese guide](./doc/Fate-allinone_deployment_guide_install_zh.md)
- Install Exchange Step By Step [Chinese guide](./doc/Fate-exchange_deployment_guide_zh.md)

thirdparty：

- Hadoop+Spark Deployment [Chinese guide](./doc/thirdparty_spark/Hadoop+Spark集群部署指南.md)

## 1.     Module Information

In a party, FATE (Federated AI Technology Enabler) has the following modules. Specific module information is as follows:

| Module Name    | Port of module | Module function                                              |
| -------------- | -------------- | ------------------------------------------------------------ |
| fate_flow      | 9360;9380      | Federated learning pipeline management module, there is only one service for each party |
| fateboard      | 8080           | Federated learning process visualization module, only one service needs to be deployed per party |
| clustermanager | 4670           | The cluster manager manages the cluster, only one instance needs to be deployed per party |
| nodemanger     | 4671           | Node manager manages the resources of each machine, each party can have multiple of this service, but a server can only have one |
| rollsite       | 9370           | Cross-site or cross-party communication components, equivalent to proxy + federation, each party has only one service |
| mysql          | 3306           | Data storage, clustermanager and fateflow dependency, each party only needs one service |

## 2. Deployment Architecture

### **2.1 Bilateral Deployment Architecture **



​                                                        Example deployment in two parties

<div style="text-align:center", align=center>
<img src="./images/arch_en.png" />
</div>



## 3. Installation Preparation

### **3.1. Server Configuration**

The following configuration information is for one-sided server configuration. If there are multiple parties, please refer to this configuration to replicate this environment:

| Server                 |                                                              |
| ---------------------- | ------------------------------------------------------------ |
| **Quantity**           | 1 or more than 1 (according to the actual server allocation module provided) |
| **Configuration**      | 8 core / 16G memory / 500G hard disk / 10M bandwidth         |
| **Operating System**   | Version: CentOS Linux release 7.2                            |
| **Dependency Package** | yum source gcc gcc-c++ make autoconfig openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-dev |
| **Users**              | User: app owner:apps (app user can sudo su root without password) |
| **File System**        | 1. The  500G hard disk is mounted to the /data directory. 2. Create /data/projects directory, projects directory belongs to app:apps. |

### 3.2 Cluster planning

| party  | partyid | hostname      | IP          | os                      | software             | services                                                |
| ------ | ------- | ------------- | ----------- | ----------------------- | -------------------- | ------------------------------------------------------- |
| PartyA | 10000   | VM_0_1_centos | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | fate, eggroll, mysql | fate_flow, fateboard, clustermanager, nodemanger, mysql |
| PartyA | 10000   | VM_0_2_centos | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | fate, eggroll        | nodemanger, rollsite                                    |
| PartyB | 9999    | VM_0_3_centos | 192.168.0.3 | CentOS 7.2/Ubuntu 16.04 | fate, eggroll, mysql | all                                                     |

### 3.3 Basic environment configuration

#### 3.3.1 hostname configuration (optional)

**1) Modify the host name**

**Run under the 192.168.0.1 root user:**

hostnamectl set-hostname VM_0_1_centos

**Run under the 192.168.0.2 root user:：**

hostnamectl set-hostname VM_0_2_centos

**Run under the 192.168.0.3 root user:**

hostnamectl set-hostname VM_0_3_centos

**2) Add Host Mapping**

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3):**

vim /etc/hosts

192.168.0.1 VM_0_1_centos

192.168.0.2 VM_0_2_centos

192.168.0.3 VM_0_3_centos

#### 3.3.2 Close selinux (optional)

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3):**

Confirm whether selinux is installed

Centos system executes: rpm -qa | grep selinux

Ubuntu system executes:  apt list --installed | grep selinux

If selinux is already installed, execute: setenforce 0

#### 3.3.3 Modify linux system parameters

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3):**

1) vim /etc/security/limits.conf

\* soft nofile 65536

\* hard nofile 65536

2) vim /etc/security/limits.d/20-nproc.conf

\* soft nproc unlimited

#### 3.3.4 Turn off the firewall (optional)

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3):**

If it is a Centos system:

systemctl disable firewalld.service

systemctl stop firewalld.service

systemctl status firewalld.service

If it is an Ubuntu system:

ufw disable

ufw status

#### 3.3.5 Software environment initialization

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

**1) create user**

```
groupadd -g 6000 apps
useradd -s /bin/bash -g apps -d /home/app app
passwd app
```

**2) Create a directory**

```
mkdir -p /data/projects/fate
mkdir -p /data/projects/install
chown -R app:apps /data/projects
```

**3) Install dependencies**

```
#centos
yum -y install gcc gcc-c++ make openssl-devel gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof sysstat telnet psmisc
#ubuntu
apt-get install -y gcc g++ make openssl supervisor libgmp-dev  libmpfr-dev libmpc-dev libaio1 libaio-dev numactl autoconf automake libtool libffi-dev libssl1.0.0 libssl-dev liblz4-1 liblz4-dev liblz4-1-dbg liblz4-tool  zlib1g zlib1g-dbg zlib1g-dev
cd /usr/lib/x86_64-linux-gnu
if [ ! -f "libssl.so.10" ];then
   ln -s libssl.so.1.0.0 libssl.so.10
   ln -s libcrypto.so.1.0.0 libcrypto.so.10
fi
```

### 3.4 Increase virtual memory

**Execute under the root user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

When used in a production environment, 128G virtual memory needs to be added due to memory calculation. Refer to:

```
cd /data
dd if=/dev/zero of=/data/swapfile128G bs=1024 count=134217728
mkswap /data/swapfile128G
swapon /data/swapfile128G
cat /proc/swaps
echo '/data/swapfile128G swap swap defaults 0 0' >> /etc/fstab
```



## 4.Project deployment


Note: The installation directory of this guide is /data/projects/install by default, the user is the app, and it should be modified according to the actual situation during installation.

### 4.1 Get the installation package

Execute under the app user of the target server (192.168.0.1 has an external network environment):

```
mkdir -p /data/projects/install
cd /data/projects/install
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/python-env-1.4.1-release.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192-linux-x64.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/mysql-1.4.1-release.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE_install_1.4.1-release.tar.gz

#Send to 192.168.0.2和192.168.0.3
scp *.tar.gz app@192.168.0.2:/data/projects/install
scp *.tar.gz app@192.168.0.3:/data/projects/install
```

### 4.2 Operating system parameter check

**Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

```
#Virtual memory, the size is not less than 128G, if it is not satisfied, please refer to #Chapter 4.6 to reset
cat /proc/swaps
Filename                                Type            Size    Used    Priority
/data/swapfile128G                      file            134217724       384     -1

#The number of file handles is not less than 65535. If it is not satisfied, please refer #to Chapter 4.3 to reset
ulimit -n
65535

#The number of user processes is not less than 64000, if it is not satisfied, please #refer to Chapter 4.3 to reset
ulimit -u
65535
```

### 4.3 Deploy mysql

**Execute under the app user of the target server (192.168.0.1 192.168.0.3)**

**1) MySQL installation:**

```
#Create mysql root directory
mkdir -p /data/projects/fate/common/mysql
mkdir -p /data/projects/fate/data/mysql

#Unzip the package
cd /data/projects/install
tar xzvf mysql-*.tar.gz
cd mysql
tar xf mysql-8.0.13.tar.gz -C /data/projects/fate/common/mysql

#Configuration settings
mkdir -p /data/projects/fate/common/mysql/mysql-8.0.13/{conf,run,logs}
cp service.sh /data/projects/fate/common/mysql/mysql-8.0.13/
cp my.cnf /data/projects/fate/common/mysql/mysql-8.0.13/conf

#initialization
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysqld --initialize --user=app --basedir=/data/projects/fate/common/mysql/mysql-8.0.13 --datadir=/data/projects/fate/data/mysql > logs/init.log 2>&1
cat logs/init.log |grep root@localhost
#Note that the root @ localhost: in the output information is the initial password of the mysql user root, which should be recorded for later changing password

#Start service
cd /data/projects/fate/common/mysql/mysql-8.0.13/
nohup ./bin/mysqld_safe --defaults-file=./conf/my.cnf --user=app >>logs/mysqld.log 2>&1 &

#Change mysql root user password
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysqladmin -h 127.0.0.1 -P 3306 -S ./run/mysql.sock -u root -p password "fate_dev"
Enter Password:【Enter the root initial password】

#Verify login
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock
Enter Password:【Enter the modified password of root: fate_dev】
```

**2）Database creation, authorization and business configuration**

```
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock
Enter Password:【fate_dev】

#Create eggroll database and tables
mysql>source /data/projects/install/mysql/create-eggroll-meta-tables.sql;

#Create fate_flow database
mysql>CREATE DATABASE IF NOT EXISTS fate_flow;

#Create remote users and authorizations
1) 192.168.0.1 execute
mysql>CREATE USER 'fate'@'192.168.0.1' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'192.168.0.1';
mysql>CREATE USER 'fate'@'192.168.0.2' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'192.168.0.2';
mysql>flush privileges;

2) 192.168.0.3 execute
mysql>CREATE USER 'fate'@'192.168.0.3' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'192.168.0.3';
mysql>flush privileges;

#insert configuration data
1) 192.168.0.1 execute
mysql>INSERT INTO server_node (host, port, node_type, status) values ('192.168.0.1', '4670', 'CLUSTER_MANAGER', 'HEALTHY');
mysql>INSERT INTO server_node (host, port, node_type, status) values ('192.168.0.1', '4671', 'NODE_MANAGER', 'HEALTHY');
mysql>INSERT INTO server_node (host, port, node_type, status) values ('192.168.0.2', '4671', 'NODE_MANAGER', 'HEALTHY');

2) 192.168.0.3 execute
mysql>INSERT INTO server_node (host, port, node_type, status) values ('192.168.0.3', '4670', 'CLUSTER_MANAGER', 'HEALTHY');
mysql>INSERT INTO server_node (host, port, node_type, status) values ('192.168.0.3', '4671', 'NODE_MANAGER', 'HEALTHY');

#check
mysql>select User,Host from mysql.user;
mysql>show databases;
mysql>use eggroll_meta;
mysql>show tables;
mysql>select * from server_node;

```



### 4.4 Deploy jdk

**Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

```
#Create jdk installation directory
mkdir -p /data/projects/fate/common/jdk
#Unzip the package
cd /data/projects/install
tar xzf jdk-8u192-linux-x64.tar.gz -C /data/projects/fate/common/jdk
cd /data/projects/fate/common/jdk
mv jdk1.8.0_192 jdk-8u192
```

### 4.5 Deploy python

**Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

```
#Create python virtual installation directory
mkdir -p /data/projects/fate/common/python

#Install miniconda3
cd /data/projects/install
tar xvf python-env-*.tar.gz
cd python-env
sh Miniconda3-4.5.4-Linux-x86_64.sh -b -p /data/projects/fate/common/miniconda3

#Install virtualenv and create virtual environment
/data/projects/fate/common/miniconda3/bin/pip install virtualenv-20.0.18-py2.py3-none-any.whl -f . --no-index

/data/projects/fate/common/miniconda3/bin/virtualenv -p /data/projects/fate/common/miniconda3/bin/python3.6 --no-wheel --no-setuptools --no-download /data/projects/fate/common/python/venv

#Install dependencies
tar xvf pip-packages-fate-*.tar.gz
source /data/projects/fate/common/python/venv/bin/activate
pip install setuptools-42.0.2-py2.py3-none-any.whl
pip install -r pip-packages-fate-1.4.1/requirements.txt -f ./pip-packages-fate-1.4.1 --no-index
pip list | wc -l
#The result should be 158
```




### 4.6 Deploy eggroll&fate

#### 4.6.1 Software deployment

```
#Software deployment
#Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)
cd /data/projects/install
tar xf FATE_install_*.tar.gz
cd FATE_install_*
tar xvf python.tar.gz -C /data/projects/fate/
tar xvf eggroll.tar.gz -C /data/projects/fate

#Execute under the app user of the target server (192.168.0.1 192.168.0.3)
tar xvf fateboard.tar.gz -C /data/projects/fate

#Set the environment variable file
#Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)
cat >/data/projects/fate/init_env.sh <<EOF
export PYTHONPATH=/data/projects/fate/python:/data/projects/fate/eggroll/python
export EGGROLL_HOME=/data/projects/fate/eggroll/
venv=/data/projects/fate/common/python/venv
source \${venv}/bin/activate
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export PATH=\$PATH:\$JAVA_HOME/bin
EOF
```

#### 4.6.2 eggroll system configuration file modification

This configuration file are shared among rollsite, clustermanager, and nodemanager, and configuration across multiple hosts on each party should be consistent. Content needs to be modified:

- Database driver, the database corresponds to the connection IP, port, user name and password used by the party. Usually the default value for the port should suffice.

  eggroll.resourcemanager.clustermanager.jdbc.driver.class.name

  eggroll.resourcemanager.clustermanager.jdbc.username

  eggroll.resourcemanager.clustermanager.jdbc.password

- Corresponding to the IP, port, nodemanager port, process tag, and port of the party clustermanager. Usually the default value for the port should suffice.

  eggroll.resourcemanager.clustermanager.host

  eggroll.resourcemanager.clustermanager.port

  eggroll.resourcemanager.nodemanager.port

  eggroll.resourcemanager.process.tag

- The Python virtual environment path, business code pythonpath, and JAVA Home path are modified. If there is no change in the related path, keep the default.

  eggroll.resourcemanager.bootstrap.egg_pair.venv

  eggroll.resourcemanager.bootstrap.egg_pair.pythonpath

  eggroll.resourcemanager.bootstrap.roll_pair_master.javahome

- Modify IP and port corresponding to the party rollsite and the party's Party Id. Default value for rollsite's port generally should suffice.

  eggroll.rollsite.host
  eggroll.rollsite.port
  eggroll.rollsite.party.id

The above parameter adjustment can be manually configured by referring to the following example, or can be completed using the following command:

Configuration file: /data/projects/fate/eggroll/conf/eggroll.properties

```
#Execute under the app user of the target server (192.168.0.1 192.168.0.2)
cat > /data/projects/fate/eggroll/conf/eggroll.properties <<EOF
[eggroll]
#db connect inf
eggroll.resourcemanager.clustermanager.jdbc.driver.class.name=com.mysql.cj.jdbc.Driver
eggroll.resourcemanager.clustermanager.jdbc.url=jdbc:mysql://192.168.0.1:3306/eggroll_meta?useSSL=false&serverTimezone=UTC&characterEncoding=utf8&allowPublicKeyRetrieval=true
eggroll.resourcemanager.clustermanager.jdbc.username=fate
eggroll.resourcemanager.clustermanager.jdbc.password=fate_dev
eggroll.data.dir=data/
eggroll.logs.dir=logs/
#clustermanager & nodemanager
eggroll.resourcemanager.clustermanager.host=192.168.0.1
eggroll.resourcemanager.clustermanager.port=4670
eggroll.resourcemanager.nodemanager.port=4671
eggroll.resourcemanager.process.tag=fate-host
eggroll.bootstrap.root.script=bin/eggroll_boot.sh
eggroll.resourcemanager.bootstrap.egg_pair.exepath=bin/roll_pair/egg_pair_bootstrap.sh
#python env
eggroll.resourcemanager.bootstrap.egg_pair.venv=/data/projects/fate/common/python/venv
#pythonpath, very import, do not modify.
eggroll.resourcemanager.bootstrap.egg_pair.pythonpath=/data/projects/fate/python:/data/projects/fate/eggroll/python
eggroll.resourcemanager.bootstrap.egg_pair.filepath=python/eggroll/roll_pair/egg_pair.py
eggroll.resourcemanager.bootstrap.roll_pair_master.exepath=bin/roll_pair/roll_pair_master_bootstrap.sh
#javahome
eggroll.resourcemanager.bootstrap.roll_pair_master.javahome=/data/projects/fate/common/jdk/jdk-8u192
eggroll.resourcemanager.bootstrap.roll_pair_master.classpath=conf/:lib/*
eggroll.resourcemanager.bootstrap.roll_pair_master.mainclass=com.webank.eggroll.rollpair.RollPairMasterBootstrap
eggroll.resourcemanager.bootstrap.roll_pair_master.jvm.options=
# for roll site. rename in the next round
eggroll.rollsite.coordinator=webank
eggroll.rollsite.host=192.168.0.2
eggroll.rollsite.port=9370
eggroll.rollsite.party.id=10000
eggroll.rollsite.route.table.path=conf/route_table.json

eggroll.session.processors.per.node=4
eggroll.session.start.timeout.ms=180000
eggroll.rollsite.adapter.sendbuf.size=1048576
eggroll.rollpair.transferpair.sendbuf.size=4150000
EOF

#Execute under the app user of the target server (192.168.0.3)
cat > /data/projects/fate/eggroll/conf/eggroll.properties <<EOF
[eggroll]
#db connect inf
eggroll.resourcemanager.clustermanager.jdbc.driver.class.name=com.mysql.cj.jdbc.Driver
eggroll.resourcemanager.clustermanager.jdbc.url=jdbc:mysql://192.168.0.3:3306/eggroll_meta?useSSL=false&serverTimezone=UTC&characterEncoding=utf8&allowPublicKeyRetrieval=true
eggroll.resourcemanager.clustermanager.jdbc.username=fate
eggroll.resourcemanager.clustermanager.jdbc.password=fate_dev
eggroll.data.dir=data/
eggroll.logs.dir=logs/
#clustermanager & nodemanager
eggroll.resourcemanager.clustermanager.host=192.168.0.3
eggroll.resourcemanager.clustermanager.port=4670
eggroll.resourcemanager.nodemanager.port=4671
eggroll.resourcemanager.process.tag=fate-guest
eggroll.bootstrap.root.script=bin/eggroll_boot.sh
eggroll.resourcemanager.bootstrap.egg_pair.exepath=bin/roll_pair/egg_pair_bootstrap.sh
#python env
eggroll.resourcemanager.bootstrap.egg_pair.venv=/data/projects/fate/common/python/venv
#pythonpath, very import, do not modify.
eggroll.resourcemanager.bootstrap.egg_pair.pythonpath=/data/projects/fate/python:/data/projects/fate/eggroll/python
eggroll.resourcemanager.bootstrap.egg_pair.filepath=python/eggroll/roll_pair/egg_pair.py
eggroll.resourcemanager.bootstrap.roll_pair_master.exepath=bin/roll_pair/roll_pair_master_bootstrap.sh
#javahome
eggroll.resourcemanager.bootstrap.roll_pair_master.javahome=/data/projects/fate/common/jdk/jdk-8u192
eggroll.resourcemanager.bootstrap.roll_pair_master.classpath=conf/:lib/*
eggroll.resourcemanager.bootstrap.roll_pair_master.mainclass=com.webank.eggroll.rollpair.RollPairMasterBootstrap
eggroll.resourcemanager.bootstrap.roll_pair_master.jvm.options=
# for roll site. rename in the next round
eggroll.rollsite.coordinator=webank
eggroll.rollsite.host=192.168.0.3
eggroll.rollsite.port=9370
eggroll.rollsite.party.id=9999
eggroll.rollsite.route.table.path=conf/route_table.json

eggroll.session.processors.per.node=4
eggroll.session.start.timeout.ms=180000
eggroll.rollsite.adapter.sendbuf.size=1048576
eggroll.rollpair.transferpair.sendbuf.size=4150000
EOF
```

#### 4.6.3 eggroll routing configuration file modification

This configuration file rollsite is used to configure routing information. You can manually configure it by referring to the following example, or you can use the following command:

Configuration file: /data/projects/fate/eggroll/conf/route_table.json

```
#Execute under the app user of the target server (192.168.0.2)
cat > /data/projects/fate/eggroll/conf/route_table.json << EOF
{
  "route_table":
  {
    "10000":
    {
      "default":[
        {
          "port": 9370,
          "ip": "192.168.0.2"
        }
      ],
      "fateflow":[
        {
          "port": 9360,
          "ip": "192.168.0.1"
        }
      ]      
    },
    "9999":
    {
      "default":[
        {
          "port": 9370,
          "ip": "192.168.0.3"
        }
      ]
    }
  },
  "permission":
  {
    "default_allow": true
  }
}
EOF

#Execute under the app user of the target server (192.168.0.3)
cat > /data/projects/fate/eggroll/conf/route_table.json << EOF
{
  "route_table":
  {
    "9999":
    {
      "default":[
        {
          "port": 9370,
          "ip": "192.168.0.3"
        }
      ],
      "fateflow":[
        {
          "port": 9360,
          "ip": "192.168.0.3"
        }
      ]      
    },
    "10000":
    {
      "default":[
        {
          "port": 9370,
          "ip": "192.168.0.2"
        }
      ]
    }
  },
  "permission":
  {
    "default_allow": true
  }
}
EOF
```

#### 4.6.4 fate dependent service configuration file modification

- fateflow

  fateflow IP , host: 192.168.0.1,guest: 192.168.0.3

​      grpc port: 9360

​      http port: 9380

- fateboard

​       fateboard IP, host: 192.168.0.1, guest: 192.168.0.3

​       fateboard port: 8080

- proxy

  proxy IP, host: 192.168.0.2, guest: 192.168.0.3---Rollsite component corresponds to IP

  proxy port：9370

  This file should be configured in json format, otherwise an error will be reported, you can refer to the following example to manually configure, you can also use the following instructions to complete.

  Configuration file: /data/projects/fate/python/arch/conf/server_conf.json

```
#Execute under the app user of the target server (192.168.0.1 192.168.0.2)
cat > /data/projects/fate/python/arch/conf/server_conf.json << EOF
{
  "servers": {
        "fateflow": {
          "host": "192.168.0.1",
          "grpc.port": 9360,
          "http.port": 9380
        },
        "fateboard": {
          "host": "192.168.0.1",
          "port": 8080
        },
        "proxy": {
          "host": "192.168.0.2",
          "port": 9370
        },
        "servings": [
          "127.0.0.1:8000"
        ]
  }
}
EOF

#Execute under the app user of the target server (192.168.0.3)
cat > /data/projects/fate/python/arch/conf/server_conf.json << EOF
{
  "servers": {
        "fateflow": {
          "host": "192.168.0.3",
          "grpc.port": 9360,
          "http.port": 9380
        },
        "fateboard": {
          "host": "192.168.0.3",
          "port": 8080
        },
        "proxy": {
          "host": "192.168.0.3",
          "port": 9370
        },
        "servings": [
          "127.0.0.1:8000"
        ]
  }
}
EOF
```

#### 4.6.5 Fate database information configuration file modification

- work_mode(1 means cluster mode, default)

- db connection IP, port, account and password

- Redis IP, port, password (no configuration required for temporary use of redis)

  This configuration file should be in yaml format, otherwise an error will be raised during parsing, you can refer to the following example to manually configure, or you can use the following command.

  Configuration file: /data/projects/fate/python/arch/conf/base_conf.yaml

```
#Execute under the app user of the target server (192.168.0.1)
cat > /data/projects/fate/python/arch/conf/base_conf.yaml <<EOF
work_mode: 1
fate_flow:
  host: 0.0.0.0
  http_port: 9380
  grpc_port: 9360
database:
  name: fate_flow
  user: fate
  passwd: fate_dev
  host: 192.168.0.1
  port: 3306
  max_connections: 100
  stale_timeout: 30
redis:
  host: 127.0.0.1
  port: 6379
  password: WEBANK_2014_fate_dev
  max_connections: 500
  db: 0
default_model_store_address:
  storage: redis
  host: 127.0.0.1
  port: 6379
  password: fate_dev
  db: 0
EOF

#Execute under the app user of the target server (192.168.0.3)
cat > /data/projects/fate/python/arch/conf/base_conf.yaml <<EOF
work_mode: 1
fate_flow:
  host: 0.0.0.0
  http_port: 9380
  grpc_port: 9360
database:
  name: fate_flow
  user: fate
  passwd: fate_dev
  host: 192.168.0.3
  port: 3306
  max_connections: 100
  stale_timeout: 30
redis:
  host: 127.0.0.1
  port: 6379
  password: WEBANK_2014_fate_dev
  max_connections: 500
  db: 0
default_model_store_address:
  storage: redis
  host: 127.0.0.1
  port: 6379
  password: fate_dev
  db: 0
EOF
```

#### 4.6.6 fateboard configuration file modification

1）application.properties

- Service port

  server.port---default

- fateflow access url

  fateflow.url, host: http://192.168.0.1:9380, guest: http://192.168.0.3:9380

- Database connection string, account number and password

  fateboard.datasource.jdbc-url, host: mysql://192.168.0.1:3306, guest: mysql://192.168.0.3:3306

  fateboard.datasource.username: fate

  fateboard.datasource.password: fate_dev

  The above parameter adjustment can be manually configured by referring to the following example, or can be completed using the following command:

  Configuration file: /data/projects/fate/fateboard/conf/application.properties

```
#Execute under the app user of the target server (192.168.0.1)
cat > /data/projects/fate/fateboard/conf/application.properties <<EOF
server.port=8080
fateflow.url=http://192.168.0.1:9380
spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver
spring.http.encoding.charset=UTF-8
spring.http.encoding.enabled=true
server.tomcat.uri-encoding=UTF-8
fateboard.datasource.jdbc-url=jdbc:mysql://192.168.0.1:3306/fate_flow?characterEncoding=utf8&characterSetResults=utf8&autoReconnect=true&failOverReadOnly=false&serverTimezone=GMT%2B8
fateboard.datasource.username=fate
fateboard.datasource.password=fate_dev
server.tomcat.max-threads=1000
server.tomcat.max-connections=20000
EOF

#Execute under the app user of the target server (192.168.0.3)
cat > /data/projects/fate/fateboard/conf/application.properties <<EOF
server.port=8080
fateflow.url=http://192.168.0.3:9380
spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver
spring.http.encoding.charset=UTF-8
spring.http.encoding.enabled=true
server.tomcat.uri-encoding=UTF-8
fateboard.datasource.jdbc-url=jdbc:mysql://192.168.0.3:3306/fate_flow?characterEncoding=utf8&characterSetResults=utf8&autoReconnect=true&failOverReadOnly=false&serverTimezone=GMT%2B8
fateboard.datasource.username=fate
fateboard.datasource.password=fate_dev
server.tomcat.max-threads=1000
server.tomcat.max-connections=20000
EOF
```

2）service.sh

```
#Execute under the app user of the target server (192.168.0.1 192.168.0.3)
cd /data/projects/fate/fateboard
vi service.sh
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
```

### 4.7 Start service

**Execute under the app user of the target server (192.168.0.2)**

```
#Start eggroll service
source /data/projects/fate/init_env.sh
cd /data/projects/fate/eggroll
sh ./bin/eggroll.sh rollsite start
sh ./bin/eggroll.sh nodemanager start
```

**Execute under the app user of the target server (192.168.0.1)**

```
#Start eggroll service
source /data/projects/fate/init_env.sh
cd /data/projects/fate/eggroll
sh ./bin/eggroll.sh clustermanager start
sh ./bin/eggroll.sh nodemanager start

#Start the fate service, fateflow depends on the start of rollsite and mysql. Make sure to start fateflow after eggroll of all nodes have been started. Otherwise, you will get stuck, and an error will be raised.

source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/fate_flow
sh service.sh start
cd /data/projects/fate/fateboard
sh service.sh start
```

**Execute under the app user of the target server (192.168.0.3)**

```
#Start eggroll service
source /data/projects/fate/init_env.sh
cd /data/projects/fate/eggroll
sh ./bin/eggroll.sh all start

#Start fate service
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/fate_flow
sh service.sh start
cd /data/projects/fate/fateboard
sh service.sh start
```

### 4.8 identify the problem

1) eggroll log

 /data/projects/fate/eggroll/logs/eggroll/bootstrap.clustermanager.err

/data/projects/fate/eggroll/logs/eggroll/clustermanager.jvm.err.log

/data/projects/fate/eggroll/logs/eggroll/nodemanager.jvm.err.log

/data/projects/fate/eggroll/logs/eggroll/bootstrap.nodemanager.err

/data/projects/fate/eggroll/logs/eggroll/bootstrap.rollsite.err

/data/projects/fate/eggroll/logs/eggroll/rollsite.jvm.err.log

2) fateflow log

/data/projects/fate/python/logs/fate_flow/

3) fateboard log

/data/projects/fate/fateboard/logs

## 5. Test

### 5.1 Toy_example deployment verification

You need to set 3 parameters for this test: guest_partyid，host_partyid，work_mode.

#### 5.1.1 Unilateral test

1) Executed on 192.168.0.1, guest_partyid and host_partyid are set to 10000:

```
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/examples/toy_example/
python run_toy_example.py 10000 10000 1
```

A result similar to the following indicates success:

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

2) Executed on 192.168.0.3, guest_partyid and host_partyid are set to 9999:

```
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/examples/toy_example/
python run_toy_example.py 9999 9999 1
```

A result similar to the following indicates success:

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

#### 5.1.2 Bilateral test

Select 9999 as the guest and execute on 192.168.0.3:

```
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/examples/toy_example/
python run_toy_example.py 9999 10000 1
```

A result similar to the following indicates success:：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"



### 5.2 Minimization testing

Start the virtual environment in host and guest respectively. Please make sure you have already uploaded the preset dataset through the provided script. 

#### 5.2.1 Upload preset Data

You can upload some preset data through one simple script easily. The script is located at /data/projects/fate/python/examples/scripts/

You can run this script as simple as running the following command:
```
python upload_default_data.py -m 1
```

For more details, please refer to [scripts' README](../examples/scripts/README.rst)

Start the virtual environment in host and guest respectively.

#### 5.2.2 Fast mode

In fast mode, min-test will start a relatively small date set (breast data set) which contains 569 lines of data.

All you need to do is just run the following command in guest party:

```
   python run_task.py -m 1 -gid 9999 -hid 10000 -aid 10000 -f fast
```

This test will automatically take breast as test data set.

There are some more parameters that you may need:

1. -f: file type. "fast" means breast data set, "normal" means default credit data set.
2. --add_sbt: if set, it will test hetero-secureboost task after testing hetero-lr.

Wait a few minutes, a result showing "success" indicates that the operation is successful.
In other cases, if FAILED or stuck, it means failure.

#### 5.2.3 Normal mode

Just replace the word "fast" with "normal" in all the commands, the rest is the same with fast mode.

### 5.3. Fateboard testing

Fateboard is a web service. Get the ip of fateboard. If fateboard service is launched successfully, you can see the task information by visiting http://${fateboard-ip}:8080.
Firewall may need to be opened. When fateboard and fatefow are deployed to separate servers, you need to specify server information of fateflow service on Fateboard page: click the gear icon on the top right corner of Board homepage -> click "add" -> fill in ip, os user, ssh, and password for fateflow service. 

## 6. System operation and maintenance

### 6.1 Service management

**Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

#### 6.1.1 Eggroll Service Management

```
source /data/projects/fate/init_env.sh
cd /data/projects/fate/eggroll
```

Start / stop / status / restart all:

```
sh ./bin/eggroll.sh all start/stop/status/restart
```

Start / stop / status / restart a single module (optional: clustermanager, nodemanager, rollsite):

```
sh ./bin/eggroll.sh clustermanager start/stop/status/restart
```

#### 6.1.2 Fate Service Management

1) Start / stop / status / restart fate_flow service

```
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/fate_flow
sh service.sh start|stop|status|restart
```

If you start module by module, you need to start eggroll first and then start fateflow. Fateflow depends on the start of eggroll.

2) Start / stop / status / restart fateboard service

```
cd /data/projects/fate/fateboard
sh service.sh start|stop|status|restart
```

#### 6.1.3 Mysql Service Management

Start / stop / status / restart mysql service

```
cd /data/projects/fate/common/mysql/mysql-8.0.13
sh ./service.sh start|stop|status|restart
```

### 6.2 View processes and ports

**Execute under the app user of the target server (192.168.0.1 192.168.0.2 192.168.0.3)**

#### 6.2.1 View process

```
#See if the process starts according to the deployment plan
ps -ef | grep -i clustermanager
ps -ef | grep -i nodemanager
ps -ef | grep -i rollsite
ps -ef | grep -i fate_flow_server.py
ps -ef | grep -i fateboard
```

#### 6.2.2 View process port

```
#Check whether the process port exists according to the deployment plan
#clustermanager
netstat -tlnp | grep 4670
#nodemanager
netstat -tlnp | grep 4671
#rollsite
netstat -tlnp | grep 9370
#fate_flow_server
netstat -tlnp | grep 9360
#fateboard
netstat -tlnp | grep 8080
```



### 6.3 Service log

| Service            | Log path                                           |
| ------------------ | -------------------------------------------------- |
| eggroll            | /data/projects/fate/eggroll/logs                   |
| fate_flow&Task log | /data/projects/fate/python/logs                    |
| fateboard          | /data/projects/fate/fateboard/logs                 |
| mysql              | /data/projects/fate/common/mysql/mysql-8.0.13/logs |

## 7. other

### 7.1 eggroll & fate package build

refer to [build guide](./build.md) 

## 7.2 Eggroll parameter tuning

Configuration file path: /data/projects/fate/eggroll/conf/eggroll.properties

Configuration file path: eggroll.session.processors.per.node

Assume that the CPU cores (cpu cores) are: c, The number of Nodemanager is: n, The number of tasks to be run simultaneously is p, then:

egg_num=eggroll.session.processors.per.node = c * 0.8 / p

partitions (Number of roll pair partitions) = egg_num * n