# 概述

一个完整的Pulsar集群由Pulsar broker、必要的 ZooKeeper 和 BookKeeper服务组成，具体的部署方式可分为两种，它们分别是：

1. 单机模式(standalone)，所有的服务在一台主机上启动，部署简单，适合用于本地开发和测试，也可用于部分生产环境。

2. 集群模式(cluster)，服务分散在不同的主机上，部署稍微复杂但具有高可用、负载均衡等特性适用于生产环境。

## 一. 前置条件

1. Java 8
2. Linux 操作系统

## 二. 下载Pulsar 2.7.0安装包

从 Pulsar 官网[下载页](https://pulsar.apache.org/download/)下载

或

使用 wget 命令下载：

``` bash
$ wget https://archive.apache.org/dist/pulsar/pulsar-2.7.0/apache-pulsar-2.7.0-bin.tar.gz
```


下载好压缩文件后，解压缩至 **"/data/projects/common"** 并使用`cd`命令进入文件所在位置：

``` bash
$ tar -C /data/projects/common -xvfz apache-pulsar-2.7.0-bin.tar.gz
$ cd /data/projects/common/apache-pulsar-2.7.0
```

## 三. 启动单机版Pulsar集群

1. 编辑"conf/standalone.conf"文件, 修改如下：

``` bash
# 修改自动删除topic功能
brokerDeleteInactiveTopicsEnabled=true

# 修改bookkie的frame容量大小位128MB，默认为5MB
nettyMaxFrameSizeBytes=134217728

# 增加pulsar message容量的大小为128MB，默认为5MB
maxMessageSize=134217728
```

2. 启动pulsar集群

``` bash
$ bin/pulsar standalone -nss 2>&1 > logs/standalone.log &
```
上述命令会把pulsar的日志导出到“logs/standalone.log”文件。

3. 停止pulsar集群

当pulsar集群不再需要时可用以下命令停止。

``` bash
$ kill $(ps -aux | grep pulsar | grep -v "grep" | awk '{print $2}')
```

当不需要pulsar集群产生的数据时可以移出除"apache-pulsar-2.7.0"下的"data"目录。

更多关于单机版Pulsar的配置，请参考Pulsar官方文档[Set up a standalone Pulsar locally](https://pulsar.apache.org/docs/zh-CN/standalone/)

## 四. 启动集群版Pulsar集群

对于集群模式的Pulsar服务部署，请参考官方文档[Deploying a multi-cluster on bare metal](https://pulsar.apache.org/docs/zh-CN/deploy-bare-metal-multi-cluster)，其中[部署配置存储集群](https://pulsar.apache.org/docs/zh-CN/deploy-bare-metal-multi-cluster/#%E9%83%A8%E7%BD%B2%E9%85%8D%E7%BD%AE%E5%AD%98%E5%82%A8%E9%9B%86%E7%BE%A4)部分的步骤可略过。
