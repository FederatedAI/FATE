#                     Fate exchange部署指南

# 1.服务器配置

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | 1                                                            |
|   配置   | 8 core /16GB memory / 500GB硬盘/10M带宽                      |
| 操作系统 | CentOS linux 7.2及以上                                       |
|  依赖包  | yum源： gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel<br /> libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy <br />snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan <br />（可以使用初始化脚本env.sh安装） |
|   用户   | 用户：app，属主：apps（app用户需可以sudo su root而无需密码） |
| 文件系统 | 1.  500G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：app:apps |

# 2.部署规划

| party        | 主机名        | IP地址      | 操作系统   |
| ------------ | ------------- | ----------- | ---------- |
| execute node | VM_0_1_centos | 192.168.0.1 | CentOS 7.2 |
| exchange     | VM_0_3_centos | 192.168.0.2 | CentOS 7.2 |

# 3.基础环境配置

## 3.1 修改Linux最大打开文件数

**在目标服务器（192.168.0.2）root用户下执行：**

vim /etc/security/limits.conf

\* soft nofile 65536

\* hard nofile 65536

## 3.2 软件环境初始化

**1）配置sudo**

**在目标服务器（192.168.0.2）root用户下执行**

vim /etc/sudoers.d/app

app ALL=(ALL) ALL

app ALL=(ALL) NOPASSWD: ALL

Defaults !env_reset

**2）配置ssh无密登录**

**a. 在目标服务器（192.168.0.1 192.168.0.2）app用户下执行**

su app

ssh-keygen -t rsa

cat \~/.ssh/id_rsa.pub \>\> /home/app/.ssh/authorized_keys

chmod 600 \~/.ssh/authorized_keys

**b.合并id_rsa_pub文件**

**在192.168.0.1 app用户下执行**

scp \~/.ssh/authorized_keys app\@192.168.0.2:/home/app/.ssh

输入app密码

**c. 在目标服务器（192.168.0.1）app用户下执行ssh 测试**

ssh app@192.168.0.2

4.项目部署
==========

注：此指导安装目录默认为/data/projects/，执行用户为app，安装时根据具体实际情况修改。

4.1 代码获取和打包
------------

**在目标服务器（192.168.0.1 具备外网环境）app用户下执行**

进入执行节点的/data/projects/目录，执行：

cd /data/projects/

git clone https://github.com/FederatedAI/FATE.git

cd FATE/cluster-deploy/scripts

bash packaging.sh 

构建好的包会放在FATE/cluster-deploy/packages目录下。

4.2 配置文件修改和示例
----------------

**在目标服务器（192.168.0.1）app用户下执行**

进入到FATE目录下的FATE/cluster-deploy/scripts目录下，修改配置文件allinone_cluster_configurations.sh.

配置文件allinone_cluster_configurations.sh说明：

| 配置项           | 配置项意义                                   | 配置项值                                                     | 说明                                                         |
| ---------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| user             | 操作用户                                     | 默认为app                                                    | 使用默认值                                                   |
| deploy_dir       | Fate安装路径                                 | 默认为 /data/projects/fate                                   | 使用默认值                                                   |
| party_list       | Party的id号                                  | 每个数组元素代表一个partyid，只支持数字，比如9999,10000.     | 只部署一个party，只填写一个partyid，部署两个party，填写两个partyid。部署exchange默认为空。 |
| node_list        | 所有party的待部署服务器列表                  | 表示所有party中包含的服务器ip列表                            | 部署一个party，只填写一个ip，部署两个party，填写两个ip。如果需要一个节点部署两个party，party_list处填写两个id，此处只填写一个IP。 |
| db_auth          | metaservice Jdbc连接数据库配置               | metaservice服务jdbc配置，填写数据库用户名和密码（此用户需要具有create database权限） | 两个party配置相同。                                          |
| redis_password   | Redis密码                                    | 默认 : fate_dev                                              | 使用默认值，两个party配置相同。                              |
| cxx_compile_flag | 用于Storage-Service-cxx节点complie方法的切换 | 默认：false                                                  | 如果服务器系统不满足Storage-Service-cxx节点的编译要求，请尝试使用true。 |

**配置示例参考：**

```
#!/bin/bash

user=app
deploy_dir=/data/projects/fate
party_list=()
node_list=(192.168.0.2)
db_auth=(fate_dev fate_dev)
redis_password=fate_dev
cxx_compile_flag=false
```

4.3 部署
--------

按照上述配置含义修改allinone_cluster_configurations.sh文件对应的配置项后，然后在FATE/cluster-deploy/scripts目录下执行部署脚本：

```
cd FATE/cluster-deploy/scripts
bash deploy_cluster_allinone.sh build jdk
bash deploy_cluster_allinone.sh build proxy
```

# 5.修改route信息

**在192.168.0.2 app用户下修改**

修改/data/projects/fate/proxy/conf/route_table.json：

```
{
    "route_table": {
        "10000": {
            "default": [
                {
                    "ip": "192.168.0.3",
                    "port": 9370
                }
            ]
        },
        "9999": {
            "default": [
                {
                    "ip": "192.168.0.4",
                    "port": 9370
                }
            ]
        }
    },
    "permission": {
        "default_allow": true
    }
}
```

**需要连接exchange的各party的proxy模块，app用户修改**

修改/data/projects/fate/proxy/conf/route_table.json部分：

```
 "default": {
            "default": [
                {
                    "ip": "192.168.0.2",
                    "port": 9370
                }
            ]
        },
```

6.启动和停止服务
================

6.1 启动服务
------------

**在目标服务器（192.168.0.2 ）app用户下执行**

```
cd /data/projects/fate/proxy
sh service.sh start
```

6.2 检查服务状态
----------------

**在目标服务器（192.168.0.2 ）app用户下执行**

```
cd /data/projects/fate/proxy
sh service.sh status
```

6.3 停止服务
------------

**在目标服务器（192.168.0.2 ）app用户下执行**

```
cd /data/projects/fate/proxy
sh service.sh stop
```




