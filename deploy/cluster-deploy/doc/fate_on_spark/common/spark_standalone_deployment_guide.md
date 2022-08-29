## spark standalone deployment guide

### 1. environment preparation

**Upload the following packages to the server**

```
1. jdk-8u192-linux-x64.tar.gz
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/jdk-8u192.tar.gz
2. spark-3.1.2-bin-hadoop3.2.tgz
wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
```

**unzip**

```bash
#If /data/projects/fate/common is not available, create a new mkdir -P /data/projects /fate/common
#Unzip spark
tar xvf spark-3.1.2-bin-hadoop3.2.tgz -C /data/projects/fate/common

#If JDK is not deployed in the current environment, execute
mkdir -p /data/projects/fate/common/jdk
#decompression
tar xzf jdk-8u192-linux-x64.tar.gz -C /data/projects/fate/common/jdk
```

**configure /etc/profile**

```bash
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_HOME=/data/projects/fate/common/spark-3.1.2-bin-hadoop3.2
export PATH=$SPARK_HOME/bin:$PATH
```

**Set spark parameter**

```
cd /data/projects/fate/common/spark-3.1.2-bin-hadoop3.2/conf
cp spark-env.sh.template spark-env.sh
#Add parameters
export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
export SPARK_MASTER_IP={Host IP}
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=9080
export SPARK_WORKER_WEBUI_PORT=9081
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
export SPARK_PID_DIR=/data/projects/fate/common/spark-3.1.2-bin-hadoop3.2/conf
```

### 2. Manage spark

- Start service
```bash
source /etc/profile
cd /data/projects/fate/common/spark-3.1.2-bin-hadoop3.2
./sbin/start-all.sh
```
- Stop service
```bash

source /etc/profile
cd /data/projects/fate/common/spark-3.1.2-bin-hadoop3.2
./sbin/stop-all.sh
```
- Master Web UI access

```
http://{ip}:9080
```

### 3. spark test
```bash
cd /data/projects/fate/common/spark-3.1.2-bin-hadoop3.2/bin
#spark shell test
./spark-shell --master spark://{Host IP}:7077
scala> var distFile = sc.textFile("/etc/profile")
scala> distFile.count()
res0: Long = 86

#pyspark test
./pyspark --master local[2]
# Import data
distFile = sc.textFile("/etc/profile")
# Count the number of rows
distFile.count()
```
If the file line count is successfully returned, the deployment is successful.
