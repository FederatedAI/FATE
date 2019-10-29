#                     Fate allinone部署指南

1.服务器配置
============

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | 2                                                            |
|   配置   | 8 core /16GB memory / 500GB硬盘/10M带宽                      |
| 操作系统 | CentOS linux 7.2及以上                                       |
|  依赖包  | yum源： gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan <br />（可以使用初始化脚本env.sh安装） |
|   用户   | 用户：app，属主：apps（app用户需可以sudo su root而无需密码） |
| 文件系统 | 1.  500G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：app:apps |

2.集群规划
==========

| party  | 主机名        | IP地址      | 操作系统   |
| ------ | ------------- | ----------- | ---------- |
| PartyA | VM_0_1_centos | 192.168.0.1 | CentOS 7.2 |
| PartyB | VM_0_2_centos | 192.168.0.2 | CentOS 7.2 |

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

4.项目部署
==========

注：此指导安装目录默认为/data/projects/，执行用户为app，安装时根据具体实际情况修改。

4.1 获取项目
------------

**在目标服务器（192.168.0.1 具备外网环境）app用户下执行**

进入执行节点的/data/projects/目录，执行：

cd /data/projects/

wget xxxxxxx

4.2 修改配置文件
----------------

**在目标服务器（192.168.0.1）app用户下执行**

进入到FATE目录下的FATE/cluster-deploy/scripts目录下，修改配置文件configurations.sh

配置文件configurations.sh说明：

| 配置项           | 配置项意义                                   | 配置项值                                                     | 说明                                                         |
| ---------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| user             | 操作用户                                     | 默认为app                                                    | 使用默认值                                                   |
| deploy_dir       | Fate安装路径                                 | 默认为 /data/projects/fate                                   | 使用默认值                                                   |
| party_list       | Party的id号                                  | 每个数组元素代表一个partyid，只支持数字，比如9999,10000.     | 只部署一个party，只填写一个partyid，部署两个party，填写两个partyid。 |
| party_ip         | 所有party的待部署服务器列表                  | 表示所有party中包含的服务器ip列表                            | 只部署一个party，只填写一个ip，部署两个party，填写两个ip。如果需要一个节点部署两个party，party_list处填写两个id，此处只填写一个IP。 |
| db_auth          | metaservice Jdbc连接数据库配置               | metaservice服务jdbc配置，填写数据库用户名和密码（此用户需要具有create database权限） | 两个party配置相同。                                          |
| redis_password   | Redis密码                                    | 默认 : fate_dev                                              | 使用默认值，两个party配置相同。                              |
| cxx_compile_flag | 用于Storage-Service-cxx节点complie方法的切换 | 默认：false                                                  | 如果服务器系统不满足Storage-Service-cxx节点的编译要求，请尝试使用true。 |

4.3 用例
--------

**1）两台主机partyA+partyB同时部署****

\#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=(10000 9999)
party_ip=(192.168.0.1 192.168.0.2)
db_auth=(fate_dev fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

**2）一台主机partyA+partyB同时部署****

\#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=(10000 9999)
party_ip=(192.168.0.1)
db_auth=(fate_dev fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

**3）只部署一个party**

\#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=(10000)
party_ip=(192.168.0.1)
db_auth=(fate_dev fate_dev)
redis_password=fate_dev
cxx_compile_flag=false

按照上述配置含义修改configurations.sh文件对应的配置项后，然后执行auto-packaging.sh脚本：

cd /data/projects/FATE/cluster-deploy/scripts

bash auto-packaging.sh

继续在FATE/cluster-deploy/scripts目录下执行部署脚本：

cd /data/projects/FATE/cluster-deploy/scripts

如果需要部署所有组件，执行：

bash deploy_allinone.sh all 

如果只部署部分组件：

bash deploy_allinone.sh fate_flow fate_board 

5.配置检查
==========

执行后可到各个目标服务器上进行检查对应模块的配置是否准确，每个模块的对应配置文件所在路径可在此配置文件下查看[cluster-deploy/doc](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/doc) 。

6.启动和停止服务
================

6.1 启动服务
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

cd /data/projects/fate

sh services.sh all start

cd /data/projects/fate/python/fate_flow

sh service.sh start

6.2 检查服务状态
----------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

查看各个服务进程是否启动成功：

cd /data/projects/fate

sh services.sh all status

cd /data/projects/fate/python/fate_flow

sh service.sh status

6.3 关机服务
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

若要关闭服务则使用：

cd /data/projects/fate

sh services.sh all stop

cd /data/projects/fate/python/fate_flow

sh service.sh stop

注：若有对单个服务进行启停操作则将上述命令中的all替换为相应的模块名称即可，若要启动fate_flow则需要到对应目录/data/projects/fate/python/fate_flow下执行sh service.sh start。

7.测试
======

7.1 单机测试
------------

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

source /data/projects/fate/venv/bin/activate

export PYTHONPATH=/data/projects/fate/python

cd \$PYTHONPATH

sh ./federatedml/test/run_test.sh

显示“ok”表示成功，显示 “FAILED”则表示失败，程序一般在一分钟内显示执行结果。

7.2 Toy_example部署验证
-----------------------

此测试您需要设置3个参数：guest_partyid，host_partyid，work_mode。

此测试只需在guest方egg节点执行，选定9999为guest方，在192.168.0.2上执行：

export PYTHONPATH = /data /projects/fat /python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/toy_example/

python run_toy_example.py 9999 10000 1

测试结果将显示在屏幕上。

7.3 最小化测试
--------------

##### **快速模式：**

在guest和host两方各任一egg节点中，根据需要在run_task.py中设置字段：guest_id，host_id，arbiter_id。

该文件在/data/projects/fate/python/examples/min_test_task /目录下。

**在Host节点192.168.0.1上运行：**

export PYTHONPATH = /data/projects/fate/python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/min_test_task /

sh run.sh host fast

从测试结果中获取“host_table”和“host_namespace”的值，并将它们作为参数传递给下述guest方命令。

**在Guest节点192.168.0.2上运行：**

export PYTHONPATH = /data/projects/fate/python

source /data/projects/fate/venv/bin/activate

cd /data/projects/fate/python/examples/min_test_task/

sh run.sh guest fast \$ {host_table} \$ {host_namespace} 

等待几分钟，看到结果显示“成功”字段，表明操作成功。在其他情况下，如果失败或卡住，则表示失败。

##### **正常模式**：

只需在命令中将“fast”替换为“normal”，其余部分与快速模式相同。

7.4. Fateboard testing
----------------------

Fateboard是一项Web服务。如果成功启动了fateboard服务，则可以通过访问http://192.168.0.1:8080和http://192.168.0.2:8080来查看任务信息。
