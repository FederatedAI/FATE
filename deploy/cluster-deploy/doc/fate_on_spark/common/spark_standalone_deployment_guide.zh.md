## spark单机版部署指南

### 1. 环境准备

**上传以下程序包到服务器上**

1. jdk-8u192-linux-x64.tar.gz
2. spark-2.4.1-bin-hadoop2.7.tar.gz

**解压**

```bash
tar xvf jdk-8u192-linux-x64.tar.gz -C /data/projects/common
tar xvf spark-2.4.1-bin-hadoop2.7.tar.gz -C /data/projects/common
```

**配置/etc/profile**

```bash
export JAVA_HOME=/data/projects/common/jdk1.8.0_192
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_HOME=/data/projects/common/spark-2.4.1-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
```


### 2. 运行spark服务
- 启动master
```bash
source /etc/profile
cd /data/projects/common/spark-2.4.1-bin-hadoop2.7 && ./sbin/start-master.sh
```
- 启动spark node服务
```bash

cd /data/projects/common/spark-2.4.1-bin-hadoop2.7 && ./sbin/start-slave.sh spark://node:7077
```
注意: node为机器名，请根据实际的机器名进行修改

### 3. spark测试
```bash
cd /data/projects/common/spark-2.4.1-bin-hadoop2.7/bin
./pyspark --master local[2]
# 导入数据
distFile = sc.textFile("/etc/profile")
# 统计行数
distFile.count()
```
如果能成功返回文件行数，即为部署成功。