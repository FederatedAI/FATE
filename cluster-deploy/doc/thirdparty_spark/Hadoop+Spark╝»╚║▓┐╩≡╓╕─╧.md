# Hadoop+Spark集群部署指南

1.集群规划
==========

| 节点名称 | 主机名        | IP地址      | 操作系统   |
|----------|---------------|-------------|------------|
| Master   | VM_0_1_centos | 192.168.0.1 | CentOS 7.2 |
| Slave1   | VM_0_2_centos | 192.168.0.2 | CentOS 7.2 |
| Slave2   | VM_0_2_centos | 192.168.0.3 | Centos 7.2 |

2.基础环境配置
==============

2.1 hostname配置
----------------

**1）修改主机名**

**在192.168.0.1 root用户下执行：**

hostnamectl set-hostname VM_0_1_centos

**在192.168.0.2 root用户下执行：**

hostnamectl set-hostname VM_0_2_centos

**在192.168.0.3 root用户下执行：**

hostnamectl set-hostname VM_0_3_centos

**2）加入主机映射**

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行：**

vim /etc/hosts

192.168.0.1 VM_0_1_centos

192.168.0.2 VM_0_2_centos

192.168.0.3 VM_0_3_centos

2.2 关闭selinux
---------------

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行：**

sed -i '/\^SELINUX/s/=.\*/=disabled/' /etc/selinux/config

setenforce 0

2.3 修改Linux最大打开文件数
---------------------------

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行：**

vim /etc/security/limits.conf

\* soft nofile 65536

\* hard nofile 65536

2.4 关闭防火墙
--------------

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行**

systemctl disable firewalld.service

systemctl stop firewalld.service

systemctl status firewalld.service

2.5初始化服务器
---------------

**1）初始化服务器**

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.1
192.168.0.3）root用户下执行**

groupadd -g 6000 apps

useradd -s /bin/bash -G apps -m app

passwd app

mkdir -p /data/projects/common/jdk

chown –R app:apps /data/app

**2）配置sudo**

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行**

vim /etc/sudoers.d/app

app ALL=(ALL) ALL

app ALL=(ALL) NOPASSWD: ALL

Defaults !env_reset

**3）配置ssh无密登录**

**在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行**

su app

ssh-keygen -t rsa

2）合并id_rsa_pub文件

**在192.168.0.1 app用户下执行**

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

chmod 600 \~/.ssh/authorized_keys

scp \~/.ssh/authorized_keys app\@192.168.0.2:/home/app/.ssh

输入密码：fate_dev

**在192.168.0.2 app用户下执行**

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

scp \~/.ssh/authorized_keys app\@192.168.0.3:/home/app/.ssh

输入密码：fate_dev

**在192.168.0.3 app用户下执行**

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

scp \~/.ssh/authorized_keys app\@192.168.0.1:/home/app/.ssh

scp \~/.ssh/authorized_keys app\@192.168.0.2:/home/app/.ssh

覆盖之前的文件

输入密码：fate_dev

**3）在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行**

ssh app\@192.168.0.1

ssh app\@192.168.0.2

ssh <app@192.168.0.3>

3.程序包准备
============

**\#上传以下程序包到服务器上**

jdk-8u192-linux-x64.tar.gz

hadoop-2.8.5.tar.gz

scala-2.11.12.tar.gz

spark-2.4.1-bin-hadoop2.7.tar.gz

zookeeper-3.4.5.tar.gz

**\#解压**

tar xvf hadoop-2.8.5.tar.gz -C /data/projects/common

tar xvf scala-2.11.12.tar.gz -C /data/projects/common

tar xvf spark-2.4.1-bin-hadoop2.7.tar.gz -C /data/projects/common

tar xvf zookeeper-3.4.5.tar.gz -C /data/projects/common

tar xvf jdk-8u192-linux-x64.tar.gz -C /data/projects/common/jdk

mv hadoop-2.8.5 hadoop

mv scala-2.11.12 scala

mv spark-2.4.1-bin-hadoop2.7 spark

mv zookeeper-3.4.5 zookeeper

**\#配置/etc/profile**

export JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192

export PATH=\$JAVA_HOME/bin:\$PATH

export HADOOP_HOME=/data/projects/common/hadoop

export PATH=\$PATH:\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin

export SPARK_HOME=/data/projects/common/spark

export PATH=\$SPARK_HOME/bin:\$PATH

4.Zookeeper集群部署
===================

**\#在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行**

cd /data/projects/common/zookeeper/conf

cat \>\> zoo.cfg \<\< EOF

tickTime=2000

initLimit=10

syncLimit=5

dataDir=/data/projects/common/zookeeper/data/zookeeper

dataLogDir=/data/projects/common/zookeeper/logs

clientPort=2181

maxClientCnxns=1000

server.1= 192.168.0.1:2888:3888

server.2= 192.168.0.2:2888:3888

server.3= 192.168.0.3:2888:3888

EOF

**\#master节点写1 slave节点依次类推**

echo 1\>\> /data/projects/common/zookeeper/data/zookeeper/myid

**\#启动**

nohup /data/projects/common/zookeeper/bin/zkServer.sh start &

5.Hadoop集群部署
================

**\#在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行**

cd /data/projects/common/hadoop/etc/hadoop

**在hadoop-env.sh、yarn-env.sh**

**加入**：export JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192

**拷贝core-site.xml、hdfs-site.xml、mapred-site.xml、yarn-site.xml**

**到/data/projects/common/Hadoop/etc/hadoop目录下，根据实际情况修改里面的IP主机名、目录等。**

**\#新建目录**

mkdir –p /data/projects/common/Hadoop/tmp

mkdir –p /data/projects/common/Hadoop/data/dfs/nn/local

mkdir –p /data/projects/common/Hadoop/data/dfs/nn/local

**\#启动**

在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行

hadoop-daemon.sh start journalnode

在192.168.0.1 app用户下执行

hdfs namenode –format

hadoop-daemon.sh start namenode

在192.168.0.2 app用户下操作

hdfs namenode –bootstrapStandby

在192.168.0.1 app用户下执行

hdfs zkfc –formatZK

在192.168.0.2 app用户下操作

hadoop-daemon.sh start namenode

在192.168.0.1 192.168.0.2 app用户下操作

hadoop-daemon.sh start zkfc

在192.168.0.1 192.168.0.2 app用户下操作

yarn-daemon.sh start resourcemanager

在192.168.0.1 192.168.0.2 192.168.0.3 app用户下操作

yarn-daemon.sh start nodemanager

在192.168.0.1 192.168.0.2 192.168.0.3 app用户下操作

hadoop-daemon.sh start datanode

**\#验证**

<http://192.168.0.1:50070>查看hadoop状态

http://192.168.0.1:8088查看yarn集群状态

6.Spark集群部署
===============

**\#在192.168.0.1 192.168.0.2 192.168.0.3 app用户下执行**

cd /data/projects/common/spark/conf

**在spark-env.sh加入**

export JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192

export SCALA_HOME=/data/projects/common/scala

export HADOOP_HOME=/data/projects/common/hadoop

export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop

export
SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://cpu-cluster/tmp/spark/event"

export HADOOP_COMMON_LIB_NATIVE_DIR=\$HADOOP_HOME/lib/native

export HADOOP_OPTS="-Djava.library.path=\$HADOOP_HOME/lib/native"

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${HADOOP_HOME}/lib/native

在slaves加入192.168.0.2 192.168.0.3

**\#启动**

/data/projects/common/spark/sbin/start-all.sh

**\#验证**

/data/projects/common/spark/bin/spark-shell --master yarn --deploy-mode client 
