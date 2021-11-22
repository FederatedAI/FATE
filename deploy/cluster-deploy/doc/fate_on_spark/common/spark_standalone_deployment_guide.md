## spark standalone deployment guide

### 1. environment preparation

**Upload the following packages to the server**

1. jdk-8u192-linux-x64.tar.gz
2. spark-2.4.1-bin-hadoop2.7.tar.gz

**unzip**

```bash
tar xvf jdk-8u192-linux-x64.tar.gz -C /data/projects/common
tar xvf spark-2.4.1-bin-hadoop2.7.tar.gz -C /data/projects/common
```

**configure/etc/profile**

```bash
export JAVA_HOME=/data/projects/common/jdk1.8.0_192
export PATH=$JAVA_HOME/bin:$PATH
export SPARK_HOME=/data/projects/common/spark-2.4.1-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
```


### 2. Run the spark service
- Start master
```bash
source /etc/profile
cd /data/projects/common/spark-2.4.1-bin-hadoop2.7 && ./sbin/start-master.sh
```
- Start the spark node service
```bash

cd /data/projects/common/spark-2.4.1-bin-hadoop2.7 && ./sbin/start-slave.sh spark://node:7077
```
Note: node is the machine name, please change it according to the actual machine name

### 3. spark test
```bash
cd /data/projects/common/spark-2.4.1-bin-hadoop2.7/bin
./pyspark --master local[2]
# Import data
distFile = sc.textFile("/etc/profile")
# Count the number of rows
distFile.count()
```
If the file line count is successfully returned, the deployment is successful.