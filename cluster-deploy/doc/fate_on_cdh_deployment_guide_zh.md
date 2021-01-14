# FATE ON CDH 部署指南

## 1.版本选择

|  组件  |  版本号                                                           |
| :------: | ------------------------------------------------------------ |
|   Fate   | 1.5.0 LTS                                         |
|   CDH   | 2.6                      |
|   JDK | 1.8                     |
|  Hadoop Client  | 2.8.5                                   |
|   Spark Client   | 2.4 |

注意：

1、1.5.0版支持最低的版本Hadoop为2.7，Spark为2.4。

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

## 6.基础环境配置

### 6.1 hostname配置(可选)

**1）修改主机名**

**在192.168.0.1 root用户下执行：**

```bash
hostnamectl set-hostname VM-0-1-centos
```

**在192.168.0.2 root用户下执行：**

```bash
hostnamectl set-hostname VM-0-2-centos
```

**2）加入主机映射**

**在目标服务器（192.168.0.1 192.168.0.2 ）root用户下执行：**

```bash
vim /etc/hosts
192.168.0.1 VM-0-1-centos
192.168.0.2 VM-0-2-centos
```

### 6.2 关闭SELinux(可选)


**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

确认是否已安装SELinux

CentOS系统执行：

```bash
rpm -qa | grep selinux
```

Ubuntu系统执行：

```bash
apt list --installed | grep selinux
```

如果已安装了SELinux就执行：

```bash
setenforce 0
```

### 6.3 修改Linux系统参数

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行：**

```bash
vim /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
```

```bash
vim /etc/security/limits.d/20-nproc.conf
* soft nproc unlimited
```

### 4.4 关闭防火墙(可选)


**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行**

如果是CentOS系统：

```bash
systemctl disable firewalld.service
systemctl stop firewalld.service
systemctl status firewalld.service
```

如果是Ubuntu系统：

```bash
ufw disable
ufw status
```

### 6.5 软件环境初始化

**在目标服务器（192.168.0.1 192.168.0.2）root用户下执行**

**1）创建用户**

```bash
groupadd -g 6000 hdfs
useradd -s /bin/bash -g hdfs -d /home/hdfs hdfs
passwd hdfs
```

**2）创建目录**

```bash
mkdir -p /data/projects/fate
mkdir -p /data/projects/install
chown -R hdfs:hdfs /data/projects
```
重要：要确保所有主机的/date/projects目录是hdfs用户所有，属主是hdfs。避免后续部署过程中出错。

**3）安装依赖**

```bash
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

## 7.项目部署

注：此指导安装目录默认为/data/projects/install，执行用户为hdfs，安装时根据具体实际情况修改。

### 7.1 获取安装包


在目标服务器（192.168.0.1 具备外网环境）hdfs用户下执行:

```bash
mkdir -p /data/projects/install
cd /data/projects/install
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/python-env-1.5.0-release.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192-linux-x64.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/mysql-1.5.0-release.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/openresty-1.17.8.2.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE_install_1.5.0_preview.tar.gz

#传输到192.168.0.1和192.168.0.2
scp *.tar.gz hdfs@192.168.0.1:/data/projects/install
scp *.tar.gz hdfs@192.168.0.2:/data/projects/install
```

### 7.2 操作系统参数检查

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**

```bash
#文件句柄数，不低于65535，如不满足需参考4.3章节重新设置
ulimit -n
65535

#用户进程数，不低于64000，如不满足需参考4.3章节重新设置
ulimit -u
65535
```

### 7.3 部署MySQL

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**

**1）MySQL安装：**

```bash
#建立mysql根目录
mkdir -p /data/projects/fate/common/mysql
mkdir -p /data/projects/fate/data/mysql

#解压缩软件包
cd /data/projects/install
tar xzvf mysql-*.tar.gz
cd mysql
tar xf mysql-8.0.13.tar.gz -C /data/projects/fate/common/mysql

#配置设置
mkdir -p /data/projects/fate/common/mysql/mysql-8.0.13/{conf,run,logs}
cp service.sh /data/projects/fate/common/mysql/mysql-8.0.13/
cp my.cnf /data/projects/fate/common/mysql/mysql-8.0.13/conf

#初始化
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysqld --initialize --user=hdfs --basedir=/data/projects/fate/common/mysql/mysql-8.0.13 --datadir=/data/projects/fate/data/mysql > logs/init.log 2>&1
cat logs/init.log |grep root@localhost
#注意输出信息中root@localhost:后的是mysql用户root的初始密码，需要记录，后面修改密码需要用到

#启动服务
cd /data/projects/fate/common/mysql/mysql-8.0.13/
nohup ./bin/mysqld_safe --defaults-file=./conf/my.cnf --user=hdfs >>logs/mysqld.log 2>&1 &

#修改mysql root用户密码
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysqladmin -h 127.0.0.1 -P 3306 -S ./run/mysql.sock -u root -p password "fate_dev"
Enter Password:【输入root初始密码】

#验证登陆
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock
Enter Password:【输入root修改后密码:fate_dev】
```

**2）建库授权和业务配置**

```bash
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock
Enter Password:【fate_dev】

#创建fate_flow库
mysql>CREATE DATABASE IF NOT EXISTS fate_flow;

#创建远程用户和授权
1) 192.168.0.1执行
mysql>CREATE USER 'fate'@'%' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'%';
mysql>flush privileges;

2) 192.168.0.2执行
mysql>CREATE USER 'fate'@'%' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'%';
mysql>flush privileges;

#校验
mysql>select User,Host from mysql.user;
mysql>show databases;
mysql>use eggroll_meta;
mysql>show tables;
mysql>select * from server_node;

```

### 7.4 部署JDK

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**:

```bash
#创建jdk安装目录
mkdir -p /data/projects/fate/common/jdk
#解压缩
cd /data/projects/install
tar xzf jdk-8u192-linux-x64.tar.gz -C /data/projects/fate/common/jdk
cd /data/projects/fate/common/jdk
mv jdk1.8.0_192 jdk-8u192
#配置profiles
sudo vi /etc/profile
export JAVA_HOME=/data/projects/common/jdk/jdk-8u192
export PATH=$JAVA_HOME/bin:$PATH
```

### 7.5 部署python

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**:

```bash
#创建python虚拟化安装目录
mkdir -p /data/projects/fate/common/python

#安装miniconda3
cd /data/projects/install
tar xvf python-env-*.tar.gz
cd python-env
sh Miniconda3-4.5.4-Linux-x86_64.sh -b -p /data/projects/fate/common/miniconda3

#安装virtualenv和创建虚拟化环境
/data/projects/fate/common/miniconda3/bin/pip install virtualenv-20.0.18-py2.py3-none-any.whl -f . --no-index

/data/projects/fate/common/miniconda3/bin/virtualenv -p /data/projects/fate/common/miniconda3/bin/python3.6 --no-wheel --no-setuptools --no-download /data/projects/fate/common/python/venv

#安装依赖包
tar xvf pip-packages-fate-*.tar.gz
source /data/projects/fate/common/python/venv/bin/activate
pip install setuptools-42.0.2-py2.py3-none-any.whl
pip install -r pip-packages-fate-1.4.2/requirements.txt -f ./pip-packages-fate-1.4.2 --no-index
pip list | wc -l
```

### 7.6 部署NginX

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**:

```bash
cd /data/projects/install
tar xzf openresty-*.tar.gz
cd openresty-*
./configure --prefix=/data/projects/fate/proxy \
                   --with-luajit \
                   --with-http_ssl_module \
                     --with-http_v2_module \
                     --with-stream \
                     --with-stream_ssl_module \
                     -j12
make && make install
```

### 7.7部署Rabbit MQ

#### 7.7.1 环境准备
1.rabbitmq-server-generic-unix-3.6.15.tar.xz

2.otp_src_19.3.tar.gz
  
#### 7.7.2 安装Erlang
1、下载Erlang源码(otp_src_19.3.tar.gz)，并解压至/data/projects/common

```tar -zxvf otp_src_19.3.tar.gz -C /data/projects/common/```

2、配置ERL_TOP

```
cd  /data/projects/common/otp_src_19.3/
export ERL_TOP=`pwd`
```

3、编译

使用以下命令编译:
```./configure --prefix=/data/projects/common/erlang
make
make install
```
#如果出现 No curses library functions found报错，则需要安装ncuress，先下载ncurses-6.0.tar.gz
```
tar -zxvf ncurses-6.0.tar.gzcd ncurses-6.0
./configure --with-shared --without-debug --without-ada --enable-overwrite  
make
make install (如果报Permission denied，则需要root权限执行)
```
4、设置环境变量

#编译完成后，设置 ERL_HOME。编辑 /etc/profile 文件，增加以下内容：

```
cat >> /etc/profile << EOF
export ERL_HOME=/data/projects/common/erlang
export PATH=$PATH:/data/projects/common/erlang/bin
```
#5、验证

#执行命令: 

```erl```

可以进入Erlang环境，则安装成功；

#### 7.7.3 安装RabbitMQ
**1、下载RabbitMq Server安装包**

解压至/data/projects/common

```
xz -d rabbitmq-server-generic-unix-3.6.15.tar.xz
tar xvf rabbitmq-server-generic-unix-3.6.15.tar -C /data/projects/common
```

**2、添加配置文件**

在/data/projects/common/rabbitmq_server-3.6.15/etc/rabbitmq/目录增加enabled_plugins、rabbitmq.config配置文件
```
cd /data/projects/common/rabbitmq_server-3.6.15/etc/rabbitmq/
vi enabled_plugins
```

#加入以下代码段

```
[rabbitmq_federation,rabbitmq_federation_management,rabbitmq_management].

```
rabbitmq.config配置从Rabbit MQ官网下载一份对应版本的即可。

**3、允许非localhost登录**

1、执行

```./usr/local/rabbitmq/rabbitmq_server-3.7.15/sbin/rabbitmq-plugins enable rabbitmq_management```


2、修改/data/projects/common/rabbitmq_server-3.6.15/ebin/rabbit.app

将：{loopback_users, [<<”guest”>>]}，
改为：{loopback_users, []}，

**4、启动RabbitMQ**

```cd /data/projects/common/rabbitmq_server-3.6.15 && ./sbin/rabbitmq-server -detached```

**5、检查端口**

```netstat -nltp |grep 5672```

**6、访问Rabbit MQ**

http://192.168.22.83:15672/

#进行用户登录，用户名/密码：guest/guest


### 7.8部署hadoop client
在192.168.0.1 192.168.0.2 hdfs用户下执行
#### 7.8.1解压
```
tar xvf hadoop-2.8.5.tar.gz -C /data/projects/common
tar xvf scala-2.11.12.tar.gz -C /data/projects/common
mv hadoop-2.8.5 hadoop
mv scala-2.11.12 scala
mv spark-2.4.1-bin-hadoop2.7 spark
mv zookeeper-3.4.5 zookeeper
```
#### 7.8.2配置profile
```
sudo vi /etc/profile
export HADOOP_HOME=/data/projects/common/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```
#### 7.8.3更新配置

从CDH集群下载客户端配置(yarn-clientconfig)，将core-site.xml、hdfs-site.xml、mapred-site.xml、yarn-site.xml四个文件替换到/data/projects/common/hadoop/etc/hadoop目录下。

注意：

1、hadoop无需启动

2、可以一个角色对应一个CDH集群，也可以独立使用CDH集群。

#### 7.8.4更新hosts

查看xml里面使用的hostname（或者是CDH所有的主机），将所有hostname对应的ip配置到/etc/hosts里。

#### 7.8.5检查

使用hdfs命令检查hadoop连接，如果能列出hdfs上面的目录，则说明配置成功。

```
hdfs dfs -ls /
```
### 7.9部署Spark client

在192.168.0.1 192.168.0.2 hdfs用户下执行

#### 7.9.1解压
```
tar xvf spark-2.4.1-bin-hadoop2.7.tar.gz -C /data/projects/common
mv spark-2.4.1-bin-hadoop2.7 spark
```
#### 7.9.2配置profile
```
sudo vi /etc/profile
export SPARK_HOME=/data/projects/common/spark/
export PATH=$SPARK_HOME/bin:$PATH
```
#### 7.9.3修改slaves
```
cd /data/projects/common/spark/conf 
cat slaves
#加入CDH集群中Spark节点的ip 
```



#### 7.9.4 修改spark-defaults
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


#### 7.9.5 修改spark-env.sh

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

#### 7.9.6 启动

```
sh /data/projects/common/spark/sbin/start-all.sh
```
#### 7.9.7 验证

```bash
cd /data/projects/common/spark/jars
hdfs dfs -mkdir -p /tmp/spark/jars
hdfs dfs -mkdir -p /tmp/spark/event
hdfs dfs -put *jar /tmp/spark/jars
/data/projects/common/spark/bin/spark-shell --master yarn --deploy-mode client 
```


### 7.10 部署FATE

#### 7.10.1 软件部署

```
#部署软件
#在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行:
cd /data/projects/install
tar xf FATE_install_*.tar.gz
cd FATE_install_*
cp -r bin /data/projects/fate/
cp -r conf /data/projects/fate/
cp fate.env /data/projects/fate/
tar xvf python.tar.gz -C /data/projects/fate/
tar xvf fateboard.tar.gz -C /data/projects/fate
tar xvf proxy.tar.gz -C /data/projects/fate
tar xvf examples.tar.gz -C /data/projects/fate

#设置环境变量文件
#在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行:
cat >/data/projects/fate/bin/init_env.sh <<EOF
export PYTHONPATH=/data/projects/fate/python
export SPARK_HOME=/data/projects/common/spark
venv=/data/projects/fate/common/python/venv
source \${venv}/bin/activate
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export PATH=\$PATH:\$JAVA_HOME/bin
EOF
```
#### 7.10.2 NginX
#### 7.10.2.1 NginX配置文件
修改配置文件:  /data/projects/fate/proxy/nginx/conf/route_table.yaml为以下内容
```
vi /data/projects/fate/proxy/nginx/conf/nginx.conf 
#user  nobody;
worker_processes  2;
error_log  logs/error.log  info;
events {
    worker_connections  1024;
}
http {
    include       mime.types;
    default_type  application/octet-stream;
    grpc_set_header Content-Type application/grpc;
    sendfile        on;
    upstream fate_cluster {
        server 192.168.0.1:9360;   # just an invalid address as a place holder
        balancer_by_lua_file 'lua/balancer.lua';
    }
    lua_package_path "/data/projects/fate/proxy/nginx/lua/?.lua;;";
    init_worker_by_lua_file 'lua/initialize.lua';
    server {
        listen       9390 http2;
        server_name  192.168.0.1;
        location / {
            access_by_lua_file 'lua/router.lua';
            grpc_pass grpc://fate_cluster;
        }
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
    }
}
```

#### 7.10.2.2  NginX路由配置文件修改

配置文件:  /data/projects/fate/proxy/nginx/conf/route_table.yaml
此配置文件NginX使用，配置路由信息，可以参考如下例子手工配置，也可以使用以下指令完成：

```
#在目标服务器（192.168.0.1,192.168.0.2）hdfs用户下修改执行
cat > /data/projects/fate/proxy/nginx/conf/route_table.yaml << EOF
default:
  proxy:
    - host: 192.168.0.2
      port: 9390
10000:
  proxy:
    - host: 192.168.0.1
      port: 9390
  fateflow:
    - host: 192.168.0.1
      port: 9360
9999:
  proxy:
    - host: 192.168.0.2
      port: 9390
  fateflow:
    - host: 192.168.0.2
      port: 9360
EOF
```
多方的route_table配置文件可以保持一致，只要将partyId下的host和port配置正确。

#### 7.10.2.3 检测

检查配置文件是否有问题
```
/data/projects/fate/proxy/nginx/nginx -t
```
#### 7.10.2.4 启动

启动Nginx
```
/data/projects/fate/proxy/sbin/nginx/nginx -c /data/projects/fate/proxy/nginx/conf/nginx.conf
```

#### 7.10.3 FATE-Board配置文件修改

1）conf/application.properties

- 服务端口

  server.port---默认

- fateflow的访问url

  fateflow.url，host：http://192.168.0.1:9380，guest：http://192.168.0.2:9380

- 数据库连接串、账号和密码

  fateboard.datasource.jdbc-url，host：mysql://192.168.0.1:3306，guest：mysql://192.168.0.2:3306

  fateboard.datasource.username：fate

  fateboard.datasource.password：fate_dev
  
  以上参数调整可以参考如下例子手工配置，也可以使用以下指令完成：
  
  配置文件：/data/projects/fate/fateboard/conf/application.properties

```
#在目标服务器（192.168.0.1）hdfs用户下修改执行
cat > /data/projects/fate/fateboard/conf/application.properties <<EOF
server.port=18080
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

#在目标服务器（192.168.0.2）hdfs用户下修改执行
cat > /data/projects/fate/fateboard/conf/application.properties <<EOF
server.port=18080
fateflow.url=http://192.168.0.2:9380
spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver
spring.http.encoding.charset=UTF-8
spring.http.encoding.enabled=true
server.tomcat.uri-encoding=UTF-8
fateboard.datasource.jdbc-url=jdbc:mysql://192.168.0.2:3306/fate_flow?characterEncoding=utf8&characterSetResults=utf8&autoReconnect=true&failOverReadOnly=false&serverTimezone=GMT%2B8
fateboard.datasource.username=fate
fateboard.datasource.password=fate_dev
server.tomcat.max-threads=1000
server.tomcat.max-connections=20000
EOF
```

2）service.sh

```
#在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下修改执行
cd /data/projects/fate/fateboard
vi service.sh
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
```

#### 5.9.5 FATE配置文件修改
 
  配置文件：/data/projects/fate/python/conf/server_conf.yaml
  
##### 运行配置
- work_mode(为1表示集群模式，默认)

- independent_scheduling_proxy(使用nginx作为Fate-Flow调度协调代理服务，FATE on Spark下需要设置为true)

- FATE-Flow的监听ip、端口

- FATE-Board的监听ip、端口

- db的连接ip、端口、账号和密码

##### 依赖服务配置
- Spark的相关配置
    - address:home为Spark home绝对路径
    - cores_per_node为Spark集群每个节点的cpu核数
    - nodes为Spark集群节点数量

- HDFS的相关配置
    - address:name_node为hdfs的namenode完整地址
    - address:path_prefix为默认存储路径前缀，若不配置则默认为/

- RabbitMQ相关配置
    - address:self为本方站点配置
    - address:$partyid为对方站点配置

- proxy相关配置，监听ip及端口

  此配置文件格式要按照yaml格式配置，不然解析报错，可以参考如下例子手工配置，也可以使用以下指令完成。

```
#在目标服务器（192.168.0.1）hdfs用户下修改执行
cat > /data/projects/fate/python/conf/server_conf.yaml <<EOF
work_mode: 1
independent_scheduling_proxy: true
use_registry: false
fateflow:
  host: 192.168.0.1
  http_port: 9380
  grpc_port: 9360
  proxy: nginx
fateboard:
  host: 192.168.0.1
  port: 18080
database:
  name: fate_flow
  user: fate
  passwd: fate_dev
  host: 192.168.0.1
  port: 3306
  max_connections: 1000
  stale_timeout: 50
fate_on_spark:
  spark:
    home: /data/projects/common/spark
    cores_per_node: 8
    nodes: 2
  hdfs:
    name_node: hdfs://nameservice1
    path_prefix: 
  rabbitmq: 
    host: 192.168.0.1
    mng_port: 15672
    port: 5672
    user: guest
    password: guest
    route_table: /data/projects/fate/conf/rabbitmq_route_table.yaml 
  nginx:
    host: 192.168.0.1
port: 9390
EOF

#在目标服务器（192.168.0.2）hdfs用户下修改执行
cat > /data/projects/fate/python/conf/server_conf.yaml <<EOF
work_mode: 1
independent_scheduling_proxy: true
use_registry: false
fateflow:
  host: 192.168.0.2
  http_port: 9380
  grpc_port: 9360
  proxy: nginx
fateboard:
  host: 192.168.0.2
  port: 18080
database:
  name: fate_flow
  user: fate
  passwd: fate_dev
  host: 192.168.0.2
  port: 3306
  max_connections: 1000
  stale_timeout: 50
fate_on_spark:
  spark:
    home: /data/projects/common/spark
    cores_per_node: 8
    nodes: 2
  hdfs:
    name_node: hdfs://nameservice1
    path_prefix: 
  rabbitmq: 
    host: 192.168.0.2
    mng_port: 15672
    port: 5672
    user: guest
    password: guest
    route_table: /data/projects/fate/conf/rabbitmq_route_table.yaml 
  nginx:
    host: 192.168.0.2
port: 9390
EOF
```
注意：

1、fateboard端口修改，改成非8080，避免与spark 产生冲突。

2、Spark作为计算引擎，使用Nginx作为跨站点调度协调代理时，fateflow下面要加上proxy:nginx。

3、hdfs.name_node为hdfs defaultFS的地址。

4、rabbitMQ的mng_port默认为15672。

5、rabbitMQ的默认用户密码为guest/guest。

### 7.11 启动服务

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**

```
#启动FATE服务，FATE-Flow依赖MySQL的启动
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/python/fate_flow
sh service.sh start
cd /data/projects/fate/fateboard
sh service.sh start
cd /data/projects/fate/proxy
./nginx/sbin/nginx -c /data/projects/fate/proxy/nginx/conf/nginx.conf
```

### 7.12 问题定位

1）FATE-Flow日志

/data/projects/fate/logs/fate_flow/

2）FATE-Board日志

/data/projects/fate/fateboard/logs

3) NginX日志

/data/projects/fate/proxy/nginx/logs

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
请参考部署python章节

#### 8.4 安装fate flow 源码
将部署Fate机器的目录/data/projects/fate/python/下的arch、fate_arch、federatedml三个文件夹拷贝到CDH集群的/data/projects/fate/python目录下

```
#登录部署Fate的机器
cd /data/projects/fate/python/
scp -r  arch fate_arch federatedml hdfs@192.168.0.x:/data/projects/fate/python/
```

## 9.测试


### 9.1 Toy_example部署验证


此测试您需要设置3个参数：`guest_partyid`, `host_partyid`, `work_mode`, `backend`

#### 9.1.1 单边测试

1）192.168.0.1上执行，guest_partyid和host_partyid都设为10000：

```bash
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/examples/toy_example/
python run_toy_example.py 10000 10000 1 -b 1
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

2）192.168.0.2上执行，guest_partyid和host_partyid都设为9999:

```
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/examples/toy_example/
python run_toy_example.py 9999 9999 1 -b 1
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

#### 9.1.2 双边测试

选定9999为guest方，在192.168.0.2上执行：

```
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/examples/toy_example/
python run_toy_example.py 9999 10000 1 -b 1
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

### 9.2 最小化测试


#### **9.2.1 上传预设数据：**

分别在192.168.0.1和192.168.0.2上执行：

```bash
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/examples/scripts/
python upload_default_data.py -m 1
```

更多细节信息，敬请参考[脚本README](../../examples/scripts/README.rst)

#### 9.2.2 快速模式

请确保guest和host两方均已分别通过给定脚本上传了预设数据。

快速模式下，最小化测试脚本将使用一个相对较小的数据集，即包含了569条数据的breast数据集。

选定9999为guest方，在192.168.0.2上执行：

```bash
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/examples/min_test_task/
python run_task.py -m 1 -gid 9999 -hid 10000 -aid 10000 -f fast -b 1
```

其他一些可能有用的参数包括：

1. -f: 使用的文件类型. "fast" 代表 breast数据集, "normal" 代表 default credit 数据集.
2. --add_sbt: 如果被设置为1, 将在运行完lr以后，启动secureboost任务，设置为0则不启动secureboost任务，不设置此参数系统默认为1。

若数分钟后在结果中显示了“success”字样则表明该操作已经运行成功了。若出现“FAILED”或者程序卡住，则意味着测试失败。

#### 9.2.3 正常模式

只需在命令中将“fast”替换为“normal”，其余部分与快速模式相同。

### 9.3. FateBoard testing

Fate-Voard是一项Web服务。如果成功启动了fateboard服务，则可以通过访问 http://192.168.0.1:8080 和 http://192.168.0.2:8080 来查看任务信息，如果有防火墙需开通。如果fateboard和fateflow没有部署再同一台服务器，需在fateboard页面设置fateflow所部署主机的登陆信息：页面右上侧齿轮按钮--》add--》填写fateflow主机ip，os用户，ssh端口，密码。

## 10.系统运维

### 10.1 服务管理

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**

####  10.1.1 FATE服务管理

1) 启动/关闭/查看/重启fate_flow服务

```bash
source /data/projects/fate/init_env.sh
cd /data/projects/fate/python/fate_flow
sh service.sh start|stop|status|restart
```

如果逐个模块启动，需要先启动eggroll再启动fateflow，fateflow依赖eggroll的启动。

2) 启动/关闭/重启FATE-Board服务

```bash
cd /data/projects/fate/fateboard
sh service.sh start|stop|status|restart
```

3) 启动/关闭/重启NginX服务

```
cd /data/projects/fate/proxy
./nginx/sbin/nginx -s reload
./nginx/sbin/nginx -s stop
```

#### 10.1.2 MySQL服务管理

启动/关闭/查看/重启MySQL服务

```bash
cd /data/projects/fate/common/mysql/mysql-8.0.13
sh ./service.sh start|stop|status|restart
```

#### 10.1.3 Rabbit MQ

启动
```bash
cd /data/projects/fate/common/rabbitmq_server-3.6.15
sh ./sbin/rabbitmq-server -detached
```
停止
```bash
cd /data/projects/fate/common/rabbitmq_server-3.6.15
sh ./sbin/rabbitmqctl shutdown
```
#### 10.1.4 Spark

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

### 10.2 查看进程和端口

**在目标服务器（192.168.0.1 192.168.0.2）hdfs用户下执行**

##### 10.2.1 查看进程

```
#根据部署规划查看进程是否启动
ps -ef | grep -i fate_flow_server.py
ps -ef | grep -i fateboard
ps -ef | grep -i nginx
ps -ef | grep -i spark
ps -ef | grep -i rabbit
```

### 10.2.2 查看进程端口

```
#根据部署规划查看进程端口是否存在
#fate_flow_server
netstat -tlnp | grep 9360
#fateboard
netstat -tlnp | grep 18080
#nginx
netstat -tlnp | grep 9390
#spark
netstat -tlnp | grep 8080
#rabbit mq
netstat -tlnp | grep 5672
```


### 10.3 服务日志

| 服务               | 日志路径                                           |
| ------------------ | -------------------------------------------------- |
| fate_flow&任务日志 | /data/projects/fate/python/logs                    |
| fateboard          | /data/projects/fate/fateboard/logs                 |
| nginx | /data/projects/fate/proxy/nginx/logs                 |
| mysql              | /data/projects/fate/common/mysql/mysql-8.0.13/logs |

## 11. 附录

### 11.1 打包构建

参见[build指导](../build.md) 

### 11.2 部署过程中的问题

#### 11.2.1 No curses library functions found

下载ncurses-6.0.tar.gz
```
tar -zxvf ncurses-6.0.tar.gzcd ncurses-6.0
./configure --with-shared --without-debug --without-ada --enable-overwrite  
make
make install (如果报Permission denied，则需要root权限执行)
```
#### 11.2.2 wxWidgets not found, wx will NOT be usable

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
#### 11.2.3 configure: error: OpenGL libraries not available
```
yum list mesa* 
yum install mesa*
yum list|grep freeglut
yum install freeglut*
```
#### 11.2.4 Federated schedule error, rpc request error: can not support coordinate proxy config None
保证conf/service_conf.xml文件下的fateflow节点下有配置proxy:nginx

#### 11.2.5 requests.exceptions.ConnectionError: HTTPConnectionPool(host='xxx.xxx', port=15672)
启动rabbitMQ	

#### 11.2.6 OSError: Prior attempt to load libhdfs failed
安装hadoop client

#### 11.2.7 ConnectionClosedByBroker: (403) 'ACCESS_REFUSED - Login was refused using authentication mechanism PLAIN. For details see the broker logfile
确保fate_flow的配置文件conf/service_conf.xml配置的rabbitMQ的用户名和密码正确。可以使用Rabbit MQ的web url :http://192.168.0.1:15672输入用户名密码来验证。

#### 11.2.8 Caused by: io.netty.channel.AbstractChannel$AnnotatedConnectException: Connection refused: fl001/192.168.0.1:35558
确保Spark client处于启动状态。

#### 11.2.9 Cannot run program "./fate/common/python/venv/bin/python": error=2, No such file or directory
确保CDH集群的所有Spark节点和DataNode节点安装了python。
#### 11.2.10 No mudule name “fate_arch”
确保CDH集群的所有Spark节点和DataNode有Fate Flow源码。
#### 11.2.11 No mudule name “xxx”
分析日志，如果有python依赖未安装，则安装即可。

#### 11.2.12OSError: HDFS connection failed

正常不会有这个问题,如果是在同一个Fate上切换了CDH。使用相同的namespace和tablename上传时，会报这个错误。

解决方法是使用不同的namespace、tablename或者删除数据库中t_storage_table_meta表中对应的记录即可。

#### 11.2.13IllegalArgumentException: Required executor memory (1024), overhead (384 MB)

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


### 11.3 问题排查
#### 11.3.1 Uplod
执行upload操作，只需要启动Fate的Mysql和fateFlow，并保证hdfs能连接且rabbit MQ启动就好。

#### 11.3.2测试hdfs联通性
在hadoop的bin目录下执行
```./hdfs dfs -ls /```
能列出hdfs的目录即为连接成功。

#### 11.3.3测试Spark联通性
在spark的bin目录下执行
```./spark-shell --master yarn --deploy-mode client ```
可以分析日志。

#### 11.3.4任务一直处于waiting状态
任务能提交成功，但是一直处于waiting状态。先判断Rabbit MQ、Spark、Fate flow和Nginx是否处于启动状态。如果服务都正常，则可以删掉fate_flow数据库，重建fate_flow库即可 。
fate_flow_server启动时会自动创建表。
