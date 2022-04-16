## spark单机版部署指南

### 1. 环境准备

**上传以下程序包到服务器上**

```
1. jdk-8u192-linux-x64.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/jdk-8u192-linux-x64.tar.gz
2. spark-2.4.1-bin-hadoop2.7.tar.gz
wget https://archive.apache.org/dist/spark/spark-2.4.1/spark-2.4.1-bin-hadoop2.7.tgz
```

**解压**

```bash
#如没有/data/projects/fate/common则新建mkdir -p /data/projects/fate/common
#解压spark
tar xvf spark-2.4.1-bin-hadoop2.7.tgz -C /data/projects/fate/common

#如当前环境没有部署jdk则执行
mkdir -p /data/projects/fate/common/jdk
#解压缩
tar xzf jdk-8u192-linux-x64.tar.gz -C /data/projects/fate/common/jdk
cd /data/projects/fate/common/jdk
mv jdk1.8.0_192 jdk-8u192
```

**配置/etc/profile**

```bash
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_HOME=/data/projects/fate/common/spark-2.4.1-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
```

**配置spark参数**

```
cd /data/projects/fate/common/spark-2.4.1-bin-hadoop2.7/conf
cp spark-env.sh.template spark-env.sh
#增加参数
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export SPARK_MASTER_IP={主机IP}
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=9080
export SPARK_WORKER_WEBUI_PORT=9081
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
export SPARK_PID_DIR=/data/projects/fate/common/spark-2.4.1-bin-hadoop2.7/conf
```

### 2. 管理spark

- 启动服务
```bash
source /etc/profile
cd /data/projects/fate/common/spark-2.4.1-bin-hadoop2.7 
./sbin/start-all.sh
```
- 停止服务

```
source /etc/profile
cd /data/projects/fate/common/spark-2.4.1-bin-hadoop2.7 
./sbin/stop-all.sh
```

- master web UI访问

```
http://{ip}:9080
```

### 3. spark测试

```bash
cd /data/projects/fate/common/spark-2.4.1-bin-hadoop2.7/bin
#spark shell测试
./spark-shell --master spark://{主机IP}:7077
scala> var distFile = sc.textFile("/etc/profile")
scala> distFile.count()
res0: Long = 86

#pyspark测试
./pyspark --master local[2]
# 导入数据
distFile = sc.textFile("/etc/profile")
# 统计行数
distFile.count()
```
如果能成功返回文件行数，即为部署成功。