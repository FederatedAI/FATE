# FATE ON CDH 部署指南

## 1.版本选择

|  组件  |  版本号                                                           |
| :------: | ------------------------------------------------------------ |
|   Fate   | 1.5.1 LTS                                         |
|   CDH   | 2.6                      |
|   JDK | 1.8                     |
|  Hadoop Client  | 2.8.5                                   |
|   Spark Client   | 2.4 |

注意：

1、1.5.1版支持最低的版本Hadoop为2.7，Spark为2.4。

2、注意保证Fate部署的JDK和CDH部署的JDK(大版本)保持一致。

## 2.服务器配置

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | >1（根据实际情况配置）                                       |
|   配置   | 8 core /16GB memory / 500GB硬盘/10M带宽                      |
| 操作系统 | CentOS linux 7.2及以上/Ubuntu 16.04 以上                     |
|  依赖包  | （参见4.5 软件环境初始化）                                   |
|   用户   | 用户：hdfs，属主：hdfs（hdfs用户需可以sudo su root而无需密码） |
| 文件系统 | 1.  500G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：hdfs:hdfs |

## 3.集群规划

| party  | partyid | 主机名        | IP地址      | 操作系统                | 安装软件    | 服务                              |
| ------ | ------- | ------------- | ----------- | ----------------------- | ----------- | --------------------------------- |
| PartyA | 10000   | VM-0-1-centos | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | fate，mysql, nginx | fateflow，fateboard，mysql，nginx |
| PartyA | 10000   | VM-0-1-centos | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 |  Spark 、HDFS |                 Spark Client、HDFS Client                  |
| PartyA | 10000   | VM-0-1-centos | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | RabbitMQ    |                                   |
| PartyB | 9999    | VM-0-2-centos | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | fate，mysql, nginx | fateflow，fateboard，mysql，nginx |
| PartyB | 9999    | VM-0-2-centos | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | Spark 、HDFS |                 Spark Client、HDFS Client                  |
| PartyB | 9999    | VM-0-2-centos | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | RabbitMQ    |                                   |

## 4.CHD集群

| CDH  | defaultFS | 基础组件  |
| ------ | ------- | ------------- |
| 01 | hdfs://nameservice1 | HDFS、Yarn、Spark、Zookeeper |
| 02 | hdfs://nameservice2 | HDFS、Yarn、Spark、Zookeeper |

Guest方和Host方可以都以一个CDH作为后台执行引擎，也可以模拟生产环境各自独立后台引擎。

## 5.组件说明

| 软件产品 | 组件      | 端口      | 说明                                                  |
| -------- | --------- | --------- | ----------------------------------------------------- |
| fate     | fate_flow | 9360;9380 | 联合学习任务流水线管理模块，每个party只能有一个此服务 |
| fate     | fateboard | 18080      | 联合学习过程可视化模块，每个party只能有一个此服务     |
| nginx    | nginx     | 9390      | 跨站点(party)调度协调代理                             |
| mysql    | mysql     | 3306      | 元数据存储                                            |
| Spark    |           |           | 计算引擎,Client模式                                              |
| HDFS     |           |           | 存储引擎,安装hadoop client无需启动                                              |
| RabbitMQ |           | 5672          | 跨站点(party)数据交换代理                             |

## 6. 部署FATE
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的1、2、3章节, 注意使用hdfs用户代替app用户

## 7. 部署CDH集群Client
### 7.1 部署hadoop client
在192.168.0.1 192.168.0.2 hdfs用户下执行
#### 7.1.1解压
```
tar xvf hadoop-2.8.5.tar.gz -C /data/projects/common
tar xvf scala-2.11.12.tar.gz -C /data/projects/common
mv hadoop-2.8.5 hadoop
mv scala-2.11.12 scala
mv spark-2.4.1-bin-hadoop2.7 spark
mv zookeeper-3.4.5 zookeeper
```
#### 7.1.2配置profile
```
sudo vi /etc/profile
export HADOOP_HOME=/data/projects/common/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```
#### 7.1.3更新配置

从CDH集群下载客户端配置(yarn-clientconfig)，将core-site.xml、hdfs-site.xml、mapred-site.xml、yarn-site.xml四个文件替换到/data/projects/common/hadoop/etc/hadoop目录下。

注意：

1、hadoop无需启动

2、可以一个角色对应一个CDH集群，也可以独立使用CDH集群。

#### 7.1.4更新hosts

查看xml里面使用的hostname（或者是CDH所有的主机），将所有hostname对应的ip配置到/etc/hosts里。

#### 7.1.5检查

使用hdfs命令检查hadoop连接，如果能列出hdfs上面的目录，则说明配置成功。

```
hdfs dfs -ls /
```

### 7.2部署Spark client

在192.168.0.1 192.168.0.2 hdfs用户下执行

#### 7.2.1解压
```
tar xvf spark-2.4.1-bin-hadoop2.7.tar.gz -C /data/projects/common
mv spark-2.4.1-bin-hadoop2.7 spark
```
#### 7.2.2配置profile
```
sudo vi /etc/profile
export SPARK_HOME=/data/projects/common/spark/
export PATH=$SPARK_HOME/bin:$PATH
```
#### 7.2.3修改slaves
```
cd /data/projects/common/spark/conf 
cat slaves
#加入CDH集群中Spark节点的ip 
```

#### 7.2.4 修改spark-defaults
```
cat spark-defaults.conf
#加入
spark.master yarn
spark.eventLog.enabled true
spark.eventLog.dir hdfs://nameservice1/tmp/spark/event
# spark.serializer org.apache.spark.serializer.KryoSerializer
# spark.driver.memory 5g
# spark.executor.extraJavaOptions -XX:+PrintGCDetails -Dkey=value -Dnumbers="one two three"
spark.yarn.jars hdfs://nameservice1/tmp/spark/jars/*.jar
```

spark.eventLog.dir和spark.yarn.jars修改为对应的hdfs DefaultFS路径。

#### 7.2.5 修改spark-env.sh

```
#在尾部加入
export JAVA_HOME=/data/projects/common/jdk/jdk-8u192
export SCALA_HOME=/data/projects/common/scala
export HADOOP_HOME=/data/projects/common/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://fate-cluster/tmp/spark/event"
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HADOOP_HOME}/lib/native
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
```
注意修改SPARK_HISTORY_OPTS的hdfs DefaultFS路径。

#### 7.2.6 启动

```
sh /data/projects/common/spark/sbin/start-all.sh
```
#### 7.2.7 验证

```bash
cd /data/projects/common/spark/jars
hdfs dfs -mkdir -p /tmp/spark/jars
hdfs dfs -mkdir -p /tmp/spark/event
hdfs dfs -put *jar /tmp/spark/jars
/data/projects/common/spark/bin/spark-shell --master yarn --deploy-mode client 
```

### 8 CDH集群修改

此部分需要修改CDH集群部署了Spark和DataNode的节点做以下操作：

#### 8.1 修改hosts
将每方安装了Fate的ip和host name配置到Spark和DataNode的节点的/etc/hosts文件下

```
vi /etc/hosts
192.168.0.1 VM-0-1-centos
```
#### 8.2 初始化目录
```
mkdir -p /data/projects/fate/common
mkdir -p /data/projects/fate/python
```
#### 8.3 安装python
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的python部署章节

#### 8.4 安装fate flow 源码
将部署Fate机器的目录/data/projects/fate/python/下的arch、fate_arch、federatedml三个文件夹拷贝到CDH集群的/data/projects/fate/python目录下

```
#登录部署Fate的机器
cd /data/projects/fate/python/
scp -r  arch fate_arch federatedml hdfs@192.168.0.x:/data/projects/fate/python/
```

## 10. FATE配置文件修改
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第4章节
其中hdfs的namenode配置为对应的

## 11. 启动
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第5章节
## 12. 问题定位
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第6章节
## 13. 测试
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第7章节
## 14.系统运维
### 14.1 FATE
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第8章节

### 14.2 Spark

启动
```bash
cd /data/projects/fate/common/spark
sh ./sbin/start-all.sh
```
如果提示输入spark节点的密码，可以使用Ctrl+c 退出

停止
```bash
cd /data/projects/fate/common/spark
sh ./sbin/stop-all.sh
```
如果提示输入spark节点的密码，可以使用Ctrl+c 退出

## 15. 附录
请参阅部署指南：[fate_on_spark_deployment_fate_zh](fate_on_spark_deployment_fate_zh.md)的第9章节

## 16. 部署过程中的问题

#### 16.1 No curses library functions found

下载ncurses-6.0.tar.gz
```
tar -zxvf ncurses-6.0.tar.gzcd ncurses-6.0
./configure --with-shared --without-debug --without-ada --enable-overwrite  
make
make install (如果报Permission denied，则需要root权限执行)
```
#### 16.2 wxWidgets not found, wx will NOT be usable

以源码包方式安装，从http://www.wxwidgets.org/downloads/下载

下载wxWidgets源码包 后解压缩并编译安装
```
bzip2 -d wxWidgets-3.0.0.tar.bz2 tar -jxvf
tar -xvf wxWidgets-3.0.0.tar
#安装依赖库： 
yum list *gtk+* yum install gtk+extra
#进入解压缩目录
./configure --with-opengl --enable-debug --enable-unicode
```
#### 16.3 configure: error: OpenGL libraries not available
```
yum list mesa* 
yum install mesa*
yum list|grep freeglut
yum install freeglut*
```
#### 16.4 Federated schedule error, rpc request error: can not support coordinate proxy config None
保证conf/service_conf.xml文件下的fateflow节点下有配置proxy:nginx

#### 16.5 requests.exceptions.ConnectionError: HTTPConnectionPool(host='xxx.xxx', port=15672)
启动rabbitMQ	

#### 16.6 OSError: Prior attempt to load libhdfs failed
安装hadoop client

#### 16.7 ConnectionClosedByBroker: (403) 'ACCESS_REFUSED - Login was refused using authentication mechanism PLAIN. For details see the broker logfile
确保fate_flow的配置文件conf/service_conf.xml配置的rabbitMQ的用户名和密码正确。可以使用Rabbit MQ的web url :http://192.168.0.1:15672输入用户名密码来验证。

#### 16.8 Caused by: io.netty.channel.AbstractChannel$AnnotatedConnectException: Connection refused: fl001/192.168.0.1:35558
确保Spark client处于启动状态。

#### 16.9 Cannot run program "./fate/common/python/venv/bin/python": error=2, No such file or directory
确保CDH集群的所有Spark节点和DataNode节点安装了python。

#### 16.10 No mudule name “fate_arch”
确保CDH集群的所有Spark节点和DataNode有Fate Flow源码。
保证spark的spark-env.sh文件配置正确。

#### 16.11 No mudule name “xxx”
分析日志，如果有python依赖未安装，则安装即可。

#### 16.12 OSError: HDFS connection failed

正常不会有这个问题,如果是在同一个Fate上切换了CDH。使用相同的namespace和tablename上传时，会报这个错误。

解决方法是使用不同的namespace、tablename或者删除数据库中t_storage_table_meta表中对应的记录即可。

#### 16.13 IllegalArgumentException: Required executor memory (1024), overhead (384 MB)

修改参数
```
#MR ApplicationMaster占用的内存量 

yarn.app.mapreduce.am.resource.mb =4g 

#单个节点上金额分配的物理内存总量 

yarn.nodemanager.resource.memory-mb=8g 

#单个任务可申请的最多物理内存量 

yarn.scheduler.maximum-allocation-mb=4g
```
重启yarn

重新下载yarn-clientconfig配置文件替换yarn-site.xml到HADOOP_HOME/etc/hadoop配置目录下。

#### 16.14 No module named 'beautifultable'
```
source /data/projects/fate/bin/init_env.sh
pip install beautifultable
```
#### 16.15 No module named 'pika'
```
source /data/projects/fate/bin/init_env.sh
pip install pika
```

#### 16.16 ImportError: cannot import name 'fs'
检查pyarrow版本，该版本需要是0.17.1
```
source /data/projects/fate/bin/init_env.sh
pip install pyarrow==0.17.1
```
如果提示要更新pip，则更新
```
pip install --upgrade pip
```

#### 16.17 Federated schedule error
查看Nginx是否启动，且日志没报错

#### 16.18 Error: Could not find or load main class org.apache.spark.deploy.yarn.ExecutorLauncher
上传Spark的Jar包
```
cd /data/projects/common/spark/jars
hdfs dfs -mkdir -p /tmp/spark/jars
hdfs dfs -mkdir -p /tmp/spark/event
hdfs dfs -put *jar /tmp/spark/jars
/data/projects/common/spark/bin/spark-shell --master yarn --deploy-mode client
```
#### 16.19 No module named ‘pyspark’
```
source /data/projects/fate/bin/init_env.sh
pip install 'pyspark'
```
或者下载pyspark-3.0.1.tar.gz离线包进行离线安装
```
wget https://files.pythonhosted.org/packages/f0/26/198fc8c0b98580f617cb03cb298c6056587b8f0447e20fa40c5b634ced77/pyspark-3.0.1.tar.gz
tar -zxvf pyspark-3.0.1.tar.gz
python setup.py install 
```

#### 16.20 io.IOException: Cannot run program "/data/projects/fate/common/python/venv/bin/python": error=13, Permission denied
```
 chmod 777 /data/projects/ -R
```

## 17 问题排查
#### 17.1 Uplod
执行upload操作，只需要启动Fate的Mysql和fateFlow，并保证hdfs能连接且rabbit MQ启动就好。

#### 17.2测试hdfs联通性
在hadoop的bin目录下执行
```./hdfs dfs -ls /```
能列出hdfs的目录即为连接成功。

#### 17.3测试Spark联通性
在spark的bin目录下执行
```./spark-shell --master yarn --deploy-mode client ```
可以分析日志。

#### 17.4任务一直处于waiting状态
任务能提交成功，但是一直处于waiting状态。先判断Rabbit MQ、Spark、Fate flow和Nginx是否处于启动状态。如果服务都正常，则可以删掉fate_flow数据库，重建fate_flow库即可 。
fate_flow_server启动时会自动创建表。
