# FATE ON Spark/CDN FATE部署指南

## 1.基础环境配置

### 1.1 hostname配置(可选)

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

### 1.2 关闭SELinux(可选)


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

### 1.3 修改Linux系统参数

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行：**

```bash
vim /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
```

```bash
vim /etc/security/limits.d/20-nproc.conf
* soft nproc unlimited
```

### 1.4 关闭防火墙(可选)


**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行**

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

### 1.5 软件环境初始化

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）root用户下执行**

**1）创建用户**

```bash
groupadd -g 6000 apps
useradd -s /bin/bash -g apps -d /home/app app
passwd app
```

**2）创建目录**

```bash
mkdir -p /data/projects/fate
mkdir -p /data/projects/install
chown -R app:apps /data/projects
```

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

## 2.部署依赖组件

注：此指导安装目录默认为/data/projects/install，执行用户为app，安装时根据具体实际情况修改。

### 2.1 获取安装包


在目标服务器（192.168.0.1 具备外网环境）app用户下执行:

```bash
mkdir -p /data/projects/install
cd /data/projects/install
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/python-env-miniconda3-4.5.4.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192-linux-x64.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/mysql-fate-8.0.13.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/openresty-1.17.8.2.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/pip-packages-fate-${version}.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE_install_${version}_release.tar.gz

#传输到192.168.0.1和192.168.0.2
scp *.tar.gz app@192.168.0.1:/data/projects/install
scp *.tar.gz app@192.168.0.2:/data/projects/install
```
注意: 当前文档需要部署的FATE version>=1.7.0
### 2.2 操作系统参数检查

**在目标服务器（192.168.0.1 192.168.0.2 192.168.0.3）app用户下执行**

```bash
#文件句柄数，不低于65535，如不满足需参考4.3章节重新设置
ulimit -n
65535

#用户进程数，不低于64000，如不满足需参考4.3章节重新设置
ulimit -u
65535
```

### 2.3 部署MySQL

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

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
./bin/mysqld --initialize --user=app --basedir=/data/projects/fate/common/mysql/mysql-8.0.13 --datadir=/data/projects/fate/data/mysql > logs/init.log 2>&1
cat logs/init.log |grep root@localhost
#注意输出信息中root@localhost:后的是mysql用户root的初始密码，需要记录，后面修改密码需要用到

#启动服务
cd /data/projects/fate/common/mysql/mysql-8.0.13/
nohup ./bin/mysqld_safe --defaults-file=./conf/my.cnf --user=app >>logs/mysqld.log 2>&1 &

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
mysql>CREATE USER 'fate'@'192.168.0.1' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'192.168.0.1';
mysql>flush privileges;

2) 192.168.0.2执行
mysql>CREATE USER 'fate'@'192.168.0.2' IDENTIFIED BY 'fate_dev';
mysql>GRANT ALL ON *.* TO 'fate'@'192.168.0.2';
mysql>flush privileges;

#校验
mysql>select User,Host from mysql.user;
mysql>show databases;
mysql>use eggroll_meta;
mysql>show tables;
mysql>select * from server_node;

```

### 2.4 部署JDK

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**:

```bash
#创建jdk安装目录
mkdir -p /data/projects/fate/common/jdk
#解压缩
cd /data/projects/install
tar xzf jdk-8u192-linux-x64.tar.gz -C /data/projects/fate/common/jdk
cd /data/projects/fate/common/jdk
mv jdk1.8.0_192 jdk-8u192
```

### 2.5 部署python

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**:

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
cd /data/projects/install
tar xvf pip-packages-fate-*.tar.gz
source /data/projects/fate/common/python/venv/bin/activate
pip install python-env/setuptools-42.0.2-py2.py3-none-any.whl
pip install -r pip-packages-fate-${version}/requirements.txt -f ./pip-packages-fate-${version} --no-index
pip list | wc -l
```

### 2.6 部署Nginx

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**:

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

### 2.7 部署RabbitMQ(和pulsar二选一)

请参阅部署指南：[RabbitMQ_deployment_guide_zh](rabbitmq_deployment_guide.zh.md)

### 2.8 部署Pulsar(和rabbitmq二选一)

请参阅部署指南：[Pulsar部署](pulsar_deployment_guide_zh.md)

## 3 部署FATE

### 3.1 软件部署

```
#部署软件
#在目标服务器（192.168.0.1 192.168.0.2）app用户下执行:
cd /data/projects/install
tar xf FATE_install_*.tar.gz
cd FATE_install_*
cp -r bin /data/projects/fate/
cp -r conf /data/projects/fate/
cp fate.env /data/projects/fate/
tar xvf python.tar.gz -C /data/projects/fate/
tar xvf fateboard.tar.gz -C /data/projects/fate
tar xvf proxy.tar.gz -C /data/projects/fate

#设置环境变量文件
#在目标服务器（192.168.0.1 192.168.0.2）app用户下执行:
cat >/data/projects/fate/bin/init_env.sh <<EOF
fate_project_base=/data/projects/fate
export FATE_PROJECT_BASE=$fate_project_base
export FATE_DEPLOY_BASE=$fate_project_base

export PYTHONPATH=/data/projects/fate/fateflow/python:/data/projects/fate/eggroll/python:/data/projects/fate/fate/python
export EGGROLL_HOME=/data/projects/fate/eggroll
export EGGROLL_LOG_LEVEL=INFO
venv=/data/projects/fate/common/python/venv
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export PATH=$PATH:$JAVA_HOME/bin
source ${venv}/bin/activate
EOF
```

### 3.2 Nginx配置文件修改
#### 3.2.1 Nginx基础配置文件修改
配置文件:  /data/projects/fate/proxy/nginx/conf/nginx.conf
此配置文件Nginx使用，配置服务基础设置以及lua代码，一般不需要修改。
若要修改，可以参考默认nginx.conf手工修改，修改完成后使用命令检测
```
/data/projects/fate/proxy/nginx/sbin/nginx -t
```

#### 3.2.2 Nginx路由配置文件修改

配置文件:  /data/projects/fate/proxy/nginx/conf/route_table.yaml
此配置文件NginX使用，配置路由信息，可以参考如下例子手工配置，也可以使用以下指令完成：

```
#在目标服务器（192.168.0.1）app用户下修改执行
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

#在目标服务器（192.168.0.2）app用户下修改执行
cat > /data/projects/fate/proxy/nginx/conf/route_table.yaml << EOF
default:
  proxy:
    - host: 192.168.0.1
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

### 3.3 FATE-Board配置文件修改

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
#在目标服务器（192.168.0.1）app用户下修改执行
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

#在目标服务器（192.168.0.2）app用户下修改执行
cat > /data/projects/fate/fateboard/conf/application.properties <<EOF
server.port=8080
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
#在目标服务器（192.168.0.1 192.168.0.2）app用户下修改执行
cd /data/projects/fate/fateboard
vi service.sh
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
```

## 4. FATE配置文件修改
 
  配置文件：/data/projects/fate/conf/service_conf.yaml
  
##### 运行配置
- FATE引擎相关配置:

```yaml
default_engines:
  computing: spark
  federation: rabbitmq #(或pulsar)
  storage: hdfs
```

- FATE-Flow的监听ip、端口

- FATE-Board的监听ip、端口

- db的连接ip、端口、账号和密码

##### 依赖服务配置

**conf/service_conf.yaml**
```yaml
fate_on_spark:
  spark:
    home:
    cores_per_node: 40
    nodes: 1
  hdfs:
    name_node: hdfs://fate-cluster
    path_prefix:
  # rabbitmq和pulsar二选一
  rabbitmq:
    host: 127.0.0.1
    mng_port: 12345
    port: 5672
    user: fate
    password: fate
    route_table:
  pulsar:
    host: 127.0.0.1
    port: 6650
    mng_port: 8080
    cluster: standalone
    tenant: fl-tenant
    topic_ttl: 5
    route_table:     



```
- Spark的相关配置
    - home为Spark home绝对路径
    - cores_per_node为Spark集群每个节点的cpu核数
    - nodes为Spark集群节点数量

- HDFS的相关配置
    - name_node为hdfs的namenode完整地址
    - path_prefix为默认存储路径前缀，若不配置则默认为/

- RabbitMQ相关配置
    - host: 主机ip
    - mng_port: 管理端口
    - port: 服务端口
    - user：管理员用户
    - password: 管理员密码
    - route_table: 路由表信息，默认为空
    
- pulsar相关配置
    - host: 主机ip
    - port: brokerServicePort
    - mng_port: webServicePort
    - cluster：集群或单机
    - tenant: 合作方需要使用同一个tenant
    - topic_ttl： 回收资源参数
    - route_table: 路由表信息，默认为空
    
**conf/rabbitmq_route_table.yaml**
```yaml
10000:
  host: 127.0.0.1
  port: 5672
9999:
  host: 127.0.0.2
  port: 5672
```

**conf/pulsar_route_table.yaml**
```yml
9999:
  # host can be a domain like 9999.fate.org
  host: 192.168.0.4
  port: 6650
  sslPort: 6651
  # set proxy address for this pulsar cluster
  proxy: ""

10000:
  # host can be a domain like 10000.fate.org
  host: 192.168.0.3
  port: 6650
  sslPort: 6651
  proxy: ""

default:
  # compose host and proxy for party that does not exist in route table
  # in this example, the host for party 8888 will be 8888.fate.org
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
  port: 6650
  sslPort: 6651
```



- proxy相关配置(ip及端口)

**conf/service_conf.yaml**
```yaml
fateflow:
  proxy: nginx
fate_on_spark:
  nginx: 
    host: 127.0.0.1
    port: 9390
```

##### spark依赖分发模式
- "conf/service_conf.yaml"
```yaml
dependent_distribution: true # 推荐使用true
```

**注意:若该配置为"true"，可忽略下面的操作**

- 依赖准备:整个fate目录拷贝到每个work节点,目录结构保持一致

- spark配置修改：spark/conf/spark-env.sh
```shell script
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
```


## 5. 启动服务

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

```
#启动FATE服务，FATE-Flow依赖MySQL的启动
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fateflow/bin
sh service.sh start
cd /data/projects/fate/fateboard
sh service.sh start
cd /data/projects/fate/proxy
./nginx/sbin/nginx -c /data/projects/fate/proxy/nginx/conf/nginx.conf
```

## 6. 问题定位

1）FATE-Flow日志

/data/projects/fate/fateflow/logs

2）FATE-Board日志

/data/projects/fate/fateboard/logs

3) NginX日志

/data/projects/fate/proxy/nginx/logs

## 7.测试


### 7.1 Toy_example部署验证


此测试您需要设置2个参数："guest-party-id", "host-party-id 10000";
此外您还需要安装fate client，下面是离线安装方式(若已经安装fate client可忽略)
```shell script
source /data/projects/fate/bin/init_env.sh
cd /data/projects/fate/fate/python/fate_client && python setup.py install
```

#### 7.1.1 单边测试

1）192.168.0.1上执行，guest_partyid和host_partyid都设为10000：

```bash
flow test toy --guest-party-id 10000 --host-party-id 10000
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

2）192.168.0.2上执行，guest_partyid和host_partyid都设为9999:

```bash
flow test toy --guest-party-id 9999 --host-party-id 9999
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"

#### 7.1.2 双边测试

选定9999为guest方，在192.168.0.2上执行：


```bash
flow test toy --guest-party-id 9999 --host-party-id 10000
```

类似如下结果表示成功：

"2020-04-28 18:26:20,789 - secure_add_guest.py[line:126] - INFO: success to calculate secure_sum, it is 1999.9999999999998"


### 7.2. FateBoard testing

Fate-Voard是一项Web服务。如果成功启动了fateboard服务，则可以通过访问 http://192.168.0.1:8080 和 http://192.168.0.2:8080 来查看任务信息，如果有防火墙需开通。如果fateboard和fateflow没有部署再同一台服务器，需在fateboard页面设置fateflow所部署主机的登陆信息：页面右上侧齿轮按钮--》add--》填写fateflow主机ip，os用户，ssh端口，密码。


## 8.系统运维
================

### 8.1 服务管理

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

####  8.1.1 FATE服务管理

1) 启动/关闭/查看/重启fate_flow服务

```bash
source /data/projects/fate/init_env.sh
cd /data/projects/fate/fateflow/bin
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

#### 8.1.2 MySQL服务管理

启动/关闭/查看/重启MySQL服务

```bash
cd /data/projects/fate/common/mysql/mysql-8.0.13
sh ./service.sh start|stop|status|restart
```

### 8.2 查看进程和端口

**在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

##### 8.2.1 查看进程

```
#根据部署规划查看进程是否启动
ps -ef | grep -i fate_flow_server.py
ps -ef | grep -i fateboard
ps -ef | grep -i nginx
```

### 8.2.2 查看进程端口

```
#根据部署规划查看进程端口是否存在
#fate_flow_server
netstat -tlnp | grep 9360
#fateboard
netstat -tlnp | grep 8080
#nginx
netstat -tlnp | grep 9390
```


### 8.3 服务日志

| 服务               | 日志路径                                           |
| ------------------ | -------------------------------------------------- |
| fate_flow&任务日志 | /data/projects/fate/fateflow/logs                    |
| fateboard          | /data/projects/fate/fateboard/logs                 |
| nginx | /data/projects/fate/proxy/nginx/logs                 |
| mysql              | /data/projects/fate/common/mysql/mysql-8.0.13/logs |

## 9. 附录

### 9.1 打包构建

参见[build指导](../build.md) 
