#                     Fate V1.0 部署指南

1.服务器配置
============

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | \>1(根据实际提供的服务器规划部署模块)                        |
|   配置   | 16 core /32GB memory / 300GB硬盘/50M带宽                     |
| 操作系统 | CentOS linux 7.2及以上                                       |
|  依赖包  | yum源： gcc gcc-c ++ make autoconfig openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-dev<br />（可以使用初始化脚本env.sh安装） |
|   用户   | 用户：app，属主：apps（app用户需可以sudo su root而无需密码） |
| 文件系统 | 1. 300G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：app:apps |

2.集群规划
==========

| 节点名称 | 主机名        | IP地址      | 操作系统   |
| -------- | ------------- | ----------- | ---------- |
| PartyA   | VM_0_1_centos | 192.168.0.1 | CentOS 7.2 |
| PartyB   | VM_0_2_centos | 192.168.0.2 | CentOS 7.2 |

3.基础环境配置
==============

3.1 hostname配置
----------------

**1）修改主机名**

**在192.168.0.1 root用户下执行：**

hostnamectl set-hostname VM_0_1_centos

**在192.168.0.2 root用户下执行：**

hostnamectl set-hostname VM_0_2_centos

**2）加入主机映射**

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

vim /etc/hosts

192.168.0.1 VM_0_1_centos

192.168.0.2 VM_0_2_centos

3.2 关闭selinux
---------------

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

sed -i '/\^SELINUX/s/=.\*/=disabled/' /etc/selinux/config

setenforce 0

3.3 修改Linux最大打开文件数
---------------------------

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

vim /etc/security/limits.conf

\* soft nofile 65536

\* hard nofile 65536

3.4 关闭防火墙
--------------

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行**

systemctl disable firewalld.service

systemctl stop firewalld.service

systemctl status firewalld.service

3.5 软件环境初始化
------------------

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

mkdir -p /data/app

cd /data/app

wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate-base.tar.gz

tar -xf fate-base .tar

cd fate-base

### 3.5.1 初始化服务器

**1）初始化服务器**

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行**

sh env.sh

chown –R app:apps /data/app

**2）配置sudo**

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行**

vim /etc/sudoers.d/app

app ALL=(ALL) ALL

app ALL=(ALL) NOPASSWD: ALL

Defaults !env_reset

**3）配置ssh无密登录**

**a. 在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

su app

ssh-keygen -t rsa

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

chmod 600 \~/.ssh/authorized_keys

**b.合并id_rsa_pub文件**

拷贝192.168.0.1的authorized_keys 到192.168.0.2
\~/.ssh目录下,追加到192.168.0.2的id_rsa.pub到authorized_keys，然后再拷贝到192.168.0.1

**在192.168.0.1 app用户下执行**

scp \~/.ssh/authorized_keys app\@192.168.0.2:/home/app/.ssh

输入密码

**在192.168.0.2 app用户下执行**

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

scp \~/.ssh/authorized_keys app\@192.168.0.1:/home/app/.ssh

覆盖之前的文件

**c. 在目标服务器（192.168.0.1 192.168.0.2）app用户下执行ssh 测试**

ssh app\@192.168.0.1

ssh app\@192.168.0.2

### 3.5.2 配置JDK

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

sh install_java.sh

### 3.5.3 配置redis

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

sh install_redis.sh

### 3.5.4 配置Python

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

sh install_py3.sh

### 3.5.5 配置mysql

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

sh install_mysql.sh

**在192.168.0.1 app用户下执行**

\$/data/projects/common/mysql/mysql-8.0.13/bin/mysql -uroot -p –S
/data/projects/common/mysql/mysql-8.0.13/mysql.sock

Enter password:(please input the original password)

\>set password='fate_dev';

\>CREATE USER 'root'\@'192.168.0.1' IDENTIFIED BY 'fate_dev';

\>GRANT ALL ON \*.\* TO 'root'\@'192.168.0.1';

\>CREATE USER 'fate_dev'\@'192.168.0.1' IDENTIFIED BY 'fate_dev';

\>GRANT ALL ON \*.\* TO 'fate_dev'\@'192.168.0.1';

\>flush privileges;

**在192.168.0.2 app用户下执行**

\$/data/projects/common/mysql/mysql-8.0.13/bin/mysql -uroot -p –S
/data/projects/common/mysql/mysql-8.0.13/mysql.sock

Enter password:(please input the original password)

\>set password='fate_dev';

\>CREATE USER 'root'\@'192.168.0.2' IDENTIFIED BY 'fate_dev';

\>GRANT ALL ON \*.\* TO 'root'\@'192.168.0.2';

\>CREATE USER 'fate_dev'\@'192.168.0.2' IDENTIFIED BY 'fate_dev';

\>GRANT ALL ON \*.\* TO 'fate_dev'\@'192.168.0.2';

\>flush privileges;

4.项目部署
==========

注：此指导安装目录默认为/data/projects/，执行用户为app，安装时根据具体实际情况修改。

4.1 获取项目
------------

**在目标服务器（192.168.0.1 具备外网环境）app用户下执行**

进入执行节点的/data/projects/目录，执行git命令从github上拉取项目：

cd /data/projects/

git clone https://github.com/WeBankFinTech/FATE.git

4.2 Maven打包
-------------

**在目标服务器（192.168.0.1）app用户下执行**

进入项目的arch目录，进行构建打包：

cd FATE/arch

mvn clean package -DskipTests

cd FATE/fate-serving

mvn clean package -DskipTests

cd FATE/fateboard

mvn clean package -DskipTests

wget
https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/third_party.tar.gz

tar -xzvf third_party.tar.gz -C
FATE/arch/eggroll/storage-service-cxx/third_party

4.3 修改配置文件
----------------

**在目标服务器（192.168.0.1）app用户下执行**

进入到FATE目录下的FATE/cluster-deploy/scripts目录下，修改配置文件configurations.sh

配置文件configurations.sh说明：

| 配置项      | 配置项意义                                      | 配置项值                                                     | 说明                                                         |
| ----------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| user        | 操作用户                                        | 默认为app                                                    | 使用默认值                                                   |
| dir         | Fate安装路径                                    | 默认为 /data/projects/fate                                   | 使用默认值                                                   |
| mysqldir    | Mysql 安装路径                                  | 默认 : /data/projects/common/mysql/mysql-8.0.13              | 使用默认值                                                   |
| javadir     | JAVA_HOME                                       | 默认 : /data/projects/common/jdk/jdk1.8.0_192                | 使用默认值                                                   |
| venvdir     | Python virtualenv安装路径                       | 默认 : /data/projects/fate/venv                              | 使用默认值                                                   |
| redispass   | Redis密码                                       | 默认 : fate_dev                                              | 使用默认值                                                   |
| redisip     | Redis所在ip主机                                 | 每个数组元素代表一个redisip                                  | 一个party 仅需要安装一个 redis ,建议和serving-service部署在同一节点上. |
| partylist   | Party的id号                                     | 每个数组元素代表一个partyid，只支持数字，比如9999,10000.     | 根据实际partyid修改                                          |
| JDBC0       | metaservice Jdbc连接数据库配置                  | metaservice服务jdbc配置：从左到右是ip dbname username password（此用户需要具有create database权限） | 如果有多party，则顺序为JDBC0，JDBC1 ...该顺序对应于partid顺序。 |
| fateflowdb0 | 对应于party中fateflow的数据源配置               | 每一party fateflow的相应数据源配置：从左到右是ip dbname username password（此用户需要具有create database权限） | 如果有多party，则为fateflowdb0，fateflowdb1 ...，对应partyid顺序。 |
| iplist      | 所有party的待部署服务器列表(exchange角色ip除外) | 表示所有party中包含的服务器ip列表（exchange角色ip除外）      | 所有party的服务器ip都需配置到此列表，重复IP只配置一次。      |
| iplist0     | 第一个party的所有服务器ip列表                   | 一个party包含的所有ip列表，重复IP只配置一次。                | 如果有多party，则顺序为iplist0，iplist1 ...顺序对应于partyid。重复IP只配置一次 |
| fateboard0  | fateboard角色IP列表                             | 表示一个party中的fateboard的ip                               | 如果有多个party，则为fateboard0，fateboard1 ...顺序对应于partyid。 |
| Cxxcompile  | 用于Storage-Service-cxx节点complie方法的切换    | 默认：false                                                  | 如果服务器系统不满足Storage-Service-cxx节点的编译要求，请尝试使用true。 |
| fedlist0    | Federation角色IP列表                            | 表示party中具有联合角色的服务器列表（当前版本中只支持一个）  | 如果有多party，则为fedlist0，fedlist1 ...顺序对应于partyid   |
| meta0       | Meta-Service角色IP列表                          | 表示party中具有meta-service角色的服务器列表（当前版本中只支持一个） | 如果有多party，则顺序为meta0，meta1 ...顺序对应于partyid     |
| proxy0      | Proxy角色IP列表                                 | 表示party中具有proxy角色的服务器列表（当前版本中只支持一个） | 如果有多party，则顺序为proxy0，proxy1 ...顺序对应于partyid   |
| roll0       | Roll角色IP列表                                  | 表示party中具有roll角色的服务器列表（当前版本只支持一个）    | 如果有多party，则顺序为roll0，roll1 ...顺序对应于partyid     |
| egglist0    | Egg角色IP列表                                   | 表示party中具有egg角色的服务器列表。                         | 如果有多party，则顺序为egglist0，egglist1 ...顺序对应于partyid |
| fllist0     | Fate_flow角色IP列表                             | 表示party中具有fate_flow角色的服务器列表（当前版本中只支持一个，需和egg角色部署在一个服务器）。 | 如果有多party，则顺序为fllist0，fllist1 ...顺序对应于partyid |
| serving0    | Serving-server角色IP列表                        | 表示party中具有serving-server角色的服务器列表（当前版本中最多只支持2个）。 | 如果有多party，则顺序为serving0，serving1 ...顺序对应于partyid。 |
| exchangeip  | Exchange角色IP列表                              | Exchange角色IP列表（当前版本中只支持一个）                   | 如果双边部署中不存在交换角色，则它可以为空，这时，双方直接相连。 当执行单边部署时，此值可以是另一方的proxy或已有的exchange ip。 |

注意：service0，serving1只在需要在线模块时才需要配置，只部署离线模块不需要配置。

4.4 用例
--------

**1）单party部署**

**Party A（configurations.sh）:**

\#!/bin/bash

user=app

dir=/data/projects/fate

mysqldir=/data/projects/common/mysql/mysql-8.0.13

javadir=/data/projects/common/jdk/jdk1.8.0_192

venvdir=/data/projects/fate/venv

redisip=(192.168.0.1)

redispass=fate_dev

partylist=(10000)

JDBC0=(192.168.0.1 eggroll_meta root fate_dev)

fateflowdb0=(192.168.0.1 fate_flow root fate_dev)

iplist=(192.168.0.1)

iplist0=(192.168.0.1)

fateboard0=(192.168.0.1)

eggautocompile=true

fedlist0=(192.168.0.1)

meta0=(192.168.0.1)

proxy0=(192.168.0.1)

roll0=(192.168.0.1)

egglist0=(192.168.0.1)

tmlist0=(192.168.0.1)

fllist0=(192.168.0.1)

serving0=(192.168.0.1)

exchangeip=

**Party B（configurations.sh）:**

\#!/bin/bash

user=app

dir=/data/projects/fate

mysqldir=/data/projects/common/mysql/mysql-8.0.13

javadir=/data/projects/common/jdk/jdk1.8.0_192

venvdir=/data/projects/fate/venv

redisip=(192.168.0.2)

redispass=fate_dev

partylist=(9999)

JDBC0=(192.168.0.2 eggroll_meta root fate_dev)

fateflowdb0=(192.168.0.2 fate_flow root fate_dev)

iplist=(192.168.0.2)

iplist0=(192.168.0.2)

fateboard0=(192.168.0.2)

eggautocompile=true

fedlist0=(192.168.0.2)

meta0=(192.168.0.2)

proxy0=(192.168.0.2)

roll0=(192.168.0.2)

egglist0=(192.168.0.2)

tmlist0=(192.168.0.2)

fllist0=(192.168.0.2)

serving0=(192.168.0.2)

exchangeip=

**2）partyA+partyB同时部署**

\#!/bin/bash

user=app

dir=/data/projects/fate

mysqldir=/data/projects/common/mysql/mysql-8.0.13

javadir=/data/projects/common/jdk/jdk1.8.0_192

venvdir=/data/projects/fate/venv

redisip=(192.168.0.1 192.168.0.2)

redispass=fate_dev

partylist=(10000 9999)

JDBC0=(192.168.0.1 eggroll_meta root fate_dev)

JDBC1=(192.168.0.2 eggroll_meta root fate_dev)

fateflowdb0=(192.168.0.1 fate_flow root fate_dev)

fateflowdb1=(192.168.0.1 fate_flow root fate_dev)

iplist=(192.168.0.1 192.168.0.2)

iplist0=(192.168.0.1 192.168.0.2)

fateboard0=(192.168.0.1)

fateboard1=(192.168.0.2)

eggautocompile=true

fedlist0=(192.168.0.1)

fedlist1=(192.168.0.2)

meta0=(192.168.0.1)

meta1=(192.168.0.2)

proxy0=(192.168.0.1)

proxy1=(192.168.0.2)

roll0=(192.168.0.1)

roll1=(192.168.0.2)

egglist0=(192.168.0.1)

egglist1=(192.168.0.2)

tmlist0=(192.168.0.1)

tmlist1=(192.168.0.2)

fllist0=(192.168.0.1)

fllist1=(192.168.0.2)

serving0=(192.168.0.1)

serving1=(192.168.0.2)

exchangeip=

按照上述配置含义修改configurations.sh文件对应的配置项后，然后执行auto-packaging.sh脚本：

cd /data/projects/FATE/cluster-deploy/scripts

bash auto-packaging.sh

继续在FATE/cluster-deploy/scripts目录下执行部署脚本：

cd /data/projects/FATE/cluster-deploy/scripts

bash auto-deploy.sh

5.配置检查
==========

执行后可到各个目标服务器上进行检查对应模块的配置是否准确，每个模块的对应配置文件所在路径可在此配置文件下查看cluster-deploy/doc。

6.启动和停止服务
================

6.1 启动服务
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

cd /data/projects/fate

sh services.sh all start

如果该服务器是serving-server节点，则还需要：

cd /data/projects/fate/serving-server

sh service.sh start

如果该服务器是fate_flow节点，则还需要：

cd /data/projects/fate/python/fate_flow

sh service.sh start

说明：若目标环境无法安装c++环境，则可将services.sh文件中的storage-serivice-cxx替换为storage-serivice再启动即可使用java版本的storage-service模块。

6.2 检查服务状态
----------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

查看各个服务进程是否启动成功：

cd /data/projects/fate

sh services.sh all status

如果该服务器是serving-server节点，则还需要：

cd /data/projects/fate/serving-server

sh service.sh status

如果该服务器是fate_flow节点，则还需要：

cd /data/projects/fate/python/fate_flow

sh service.sh status

6.3 关机服务
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

若要关闭服务则使用：

cd /data/projects/fate

sh services.sh all stop

如果该服务器是serving-server节点，则还需要：

cd /data/projects/fate/serving-server

sh service.sh stop

如果该服务器是fate_flow节点，则还需要：

cd /data/projects/fate/python/fate_flow

sh service.sh stop

注：若有对单个服务进行启停操作则将上述命令中的all替换为相应的模块名称即可，若要启动fate_flow和Serving-server两个服务则需要到对应目录/data/projects/fate/python/fate_flow和/data/projects/fate/serving-server下执行sh
service.sh start。

7.测试
======

7.1 单机测试
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

source /data/projects/fate/venv/bin/activate

export PYTHONPATH=/data/projects/fate/python

cd \$PYTHONPATH

sh ./federatedml/test/run_test.sh

请参阅“确定”字段以指示操作成功。在其他情况下，如果失败或卡住，则表示失败，程序应在一分钟内生成结果。

7.2 Toy_example部署验证
-----------------------

要运行测试，您需要设置3个参数：guest_partyid，host_partyid，work_mode。

对于独立版本：

work_mode为0. guest_partyid和host_partyid应该相同，并且对应于运行测试的partyid。

对于分布式版本：

work_mode为1，guest_partyid和host_partyid应对应于您的分布式版本设置。
请注意分发版测试只在guest 9999:192.168.0.2进行

将不同版本的正确值传递给以下命令，然后运行：

export PYTHONPATH = /data /projects/fat /python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/toy_example/

python run_toy_example.py 9999 10000 1

测试结果将显示在屏幕上。

7.3 最小化测试
--------------

快速模式

在guest和host部分的节点中，根据需要在run_task.py中设置字段：guest_id，host_id，arbiter_id。

该文件位于/data/projects/fate/python/examples/min_test_task /中。

**在Host的节点192.168.0.1中，运行：**

export PYTHONPATH = /data/projects/fate/python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/min_test_task /

sh run.sh host fast

从测试结果中获取“host_table”和“host_namespace”的值，并将它们传递给以下命令。

**在Guest的节点：192.168.0.2中，运行：**

export PYTHONPATH = /data/projects/fate/python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/min_test_task/

sh run.sh guest fast \$ {host_table} \$ {host_namespace}

等待几分钟，看到结果显示“成功”字段，表明操作成功。在其他情况下，如果失败或卡住，则表示失败。

正常模式

只需在所有命令中将“fast”替换为“normal”，其余部分与快速模式相同。

7.4. Fateboard testing
----------------------

Fateboard是一项Web服务。如果成功启动了fateboard服务，则可以通过访问http://192.168.0.1:8080和http://192.168.0.2:8080来查看任务信息。
