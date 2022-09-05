# Hadoop+Spark Cluster Deploy Guide

## 1. Example Cluster Information

| Node Name| Hostname      | IP Address   | OS         |
| -------- | ------------- | ----------- | ---------- |
| Master   | VM-0-1-centos | 192.168.0.1 | CentOS 7.2 |
| Slave1   | VM-0-2-centos | 192.168.0.2 | CentOS 7.2 |
| Slave2   | VM-0-3-centos | 192.168.0.3 | Centos 7.2 |

## 2.Base Environment Configuration

### 2.1 Hostname Format

**1) Modify Hostname (Underscore "\_" is not allowed in hostname)**

**Execute on 192.168.0.1 as the root user:**

```bash
hostnamectl set-hostname VM-0-1-centos
```

**Execute on 192.168.0.2 as the root user:**

```bash
hostnamectl set-hostname VM-0-2-centos
```

**Execute on 192.168.0.3 as the root user:**


```bash
hostnamectl set-hostname VM-0-3-centos
```

**2) Add Host Mapping**

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
vim /etc/hosts
192.168.0.1 VM-0-1-centos
192.168.0.2 VM-0-2-centos
192.168.0.3 VM-0-3-centos
```

### 2.2 Disable SELinux

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
sed -i '/^SELINUX/s/=.*/=disabled/' /etc/selinux/config
setenforce 0
```

### 2.3 Set Linux resource limit

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
vim /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
```

### 2.4 Disable & Stop Firewall

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
systemctl disable firewalld.service
systemctl stop firewalld.service
systemctl status firewalld.service
```

### 2.5 Initialize Server

**1) Initialize Server**

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
groupadd -g 6000 apps
useradd -s /bin/bash -G apps -m app
passwd app
mkdir -p /data/projects/common/jdk
chown –R app:apps /data/projects
```

**2) Enable sudo**

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as the root user:**

```bash
vim /etc/sudoers.d/app

app ALL=(ALL) ALL
app ALL=(ALL) NOPASSWD: ALL
Defaults !env_reset
```

**3) Setup Passwordless SSH Login**

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:**

```bash
su app
ssh-keygen -t rsa
```

**Copy id_rsa.pub file**

**Execute on 192.168.0.1 as user app:**

```bash
cat ~/.ssh/id_rsa.pub >> /home/app/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
scp ~/.ssh/authorized_keys app@192.168.0.2:/home/app/.ssh
```

Enter password: fate_dev

**Execute on 192.168.0.2 as user app:**

```bash
cat ~/.ssh/id_rsa.pub >> /home/app/.ssh/authorized_keys
scp ~/.ssh/authorized_keys app@192.168.0.3:/home/app/.ssh
```

Enter password: fate_dev

**Execute on 192.168.0.3 as user app:**

```bash
cat ~/.ssh/id_rsa.pub >> /home/app/.ssh/authorized_keys
scp ~/.ssh/authorized_keys app@192.168.0.1:/home/app/.ssh
scp ~/.ssh/authorized_keys app@192.168.0.2:/home/app/.ssh
```

Overwrite previous file

Enter password: fate_dev

**Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:**

```bash
ssh app@192.168.0.1
ssh app@192.168.0.2
ssh app@192.168.0.3
```

## 3. Prepare Packages

**Upload below Packages to Servers**

1. wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/jdk-8u192.tar.gz
2. wget https://archive.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz
3. wget https://downloads.lightbend.com/scala/2.12.10/scala-2.12.10.tgz
4. wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
5. wget https://archive.apache.org/dist/zookeeper/zookeeper-3.4.14/zookeeper-3.4.14.tar.gz

**Extract**

```bash
tar xvf hadoop-3.2.0.tar.gz -C /data/projects/common
tar xvf scala-2.12.10.tgz -C /data/projects/common
tar xvf spark-3.1.2-bin-hadoop3.2.tgz -C /data/projects/common
tar xvf zookeeper-3.4.14.tar.gz -C /data/projects/common
tar xvf jdk-8u192-linux-x64.tar.gz -C /data/projects/common/jdk
mv hadoop-3.2.0 hadoop
mv scala-2.12.10 scala
mv spark-3.1.2-bin-hadoop3.2 spark
mv zookeeper-3.4.14 zookeeper
```

**Configure /etc/profile**

```bash
export JAVA_HOME=/data/projects/common/jdk/jdk-8u192
export PATH=$JAVA_HOME/bin:$PATH
export HADOOP_HOME=/data/projects/common/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export SPARK_HOME=/data/projects/common/spark
export PATH=$SPARK_HOME/bin:$PATH
```

## 4.Zookeeper Cluster Deploy

**\#Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:**

```bash
cd /data/projects/common/zookeeper/conf
cat >> zoo.cfg << EOF
> tickTime=2000
> initLimit=10
> syncLimit=5
> dataDir=/data/projects/common/zookeeper/data/zookeeper
> dataLogDir=/data/projects/common/zookeeper/logs
> clientPort=2181
> maxClientCnxns=1000
> server.1= 192.168.0.1:2888:3888
> server.2= 192.168.0.2:2888:3888
> server.3= 192.168.0.3:2888:3888
> EOF
```

**\#master node to 1 slave node**

```bash
echo 1>> /data/projects/common/zookeeper/data/zookeeper/myid
```

**\#Start**

```bash
nohup /data/projects/common/zookeeper/bin/zkServer.sh start &
```

## 5.Hadoop Cluster Deploy

**\#Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:**

```bash
cd /data/projects/common/hadoop/etc/hadoop
```

**In hadoop-env.sh、yarn-env.sh**

**Add**: export JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192

**In /data/projects/common/Hadoop/etc/hadoop change `core-site.xml`, `hdfs-site.xml`, `mapred-site.xml`, `yarn-site.xml` configuration; change IP hostname & path depending on actual environment. Please refer below for an example**

- core-site.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/data/projects/common/hadoop/tmp</value>
    </property>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://fate-cluster</value>
    </property>
    <property>
        <name>io.compression.codecs</name>
        <value>org.apache.hadoop.io.compress.GzipCodec,
            org.apache.hadoop.io.compress.DefaultCodec,
            org.apache.hadoop.io.compress.BZip2Codec,
            org.apache.hadoop.io.compress.SnappyCodec
        </value>
    </property>
    <property>
        <name>hadoop.proxyuser.root.hosts</name>
        <value>*</value>
    </property>
    <property>
        <name>hadoop.proxyuser.root.groups</name>
        <value>*</value>
    </property>
    <property>
        <name>ha.zookeeper.quorum</name>
        <value>192.168.0.1:2181,192.168.0.2:2181,192.168.0.3:2181</value>
    </property>
    <!-- Authentication for Hadoop HTTP web-consoles -->
        <property>
                <name>hadoop.http.filter.initializers</name>
                <value>org.apache.hadoop.security.AuthenticationFilterInitializer</value>
        </property>
        <property>
                <name>hadoop.http.authentication.type</name>
                <value>simple</value>
        </property>
        <property>
                <name>hadoop.http.authentication.token.validity</name>
                <value>3600</value>
        </property>
        <property>
                <name>hadoop.http.authentication.signature.secret.file</name>
                <value>/data/projects/commom/hadoop/etc/hadoop/hadoop-http-auth-signature-secret</value>
        </property>
        <property>
                <name>hadoop.http.authentication.cookie.domain</name>
                <value></value>
        </property>
        <property>
                <name>hadoop.http.authentication.simple.anonymous.allowed</name>
                <value>true</value>
        </property>
</configuration>

```

- hdfs-site.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>
    <property>
        <name>dfs.permissions.enabled</name>
        <value>false</value>
    </property>
    <property>
        <name>dfs.nameservices</name>
        <value>fate-cluster</value>
    </property>
    <property>
        <name>dfs.ha.namenodes.fate-cluster</name>
        <value>nn1,nn2</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address.fate-cluster.nn1</name>
        <value>192.168.0.1:9000</value>
    </property>
    <property>
        <name>dfs.namenode.http-address.fate-cluster.nn1</name>
        <value>192.168.0.1:50070</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address.fate-cluster.nn2</name>
        <value>192.168.0.2:9000</value>
    </property>
    <property>
        <name>dfs.namenode.http-address.fate-cluster.nn2</name>
        <value>192.168.0.2:50070</value>
    </property>
    <property>
        <name>dfs.namenode.shared.edits.dir</name>
        <value>qjournal://192.168.0.1:8485;192.168.0.2:8485;192.168.0.3:8485/fate-cluster</value>
    </property>
    <property>
        <name>dfs.journalnode.edits.dir</name>
        <value>/data/projects/common/hadoop/data/journaldata</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///data/projects/common/hadoop/data/dfs/nn/local</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/data/projects/common/hadoop/data/dfs/dn/local</value>
    </property>
    <property>
        <name>dfs.client.failover.proxy.provider.fate-cluster</name>
        <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
    </property>
    <property>
        <name>dfs.ha.fencing.methods</name>
        <value>shell(/bin/true)</value>
    </property>
    <property>
        <name>dfs.ha.fencing.ssh.private-key-files</name>
        <value>/home/app/.ssh/id_rsa</value>
    </property>
    <property>
        <name>dfs.ha.fencing.ssh.connect-timeout</name>
        <value>10000</value>
    </property>
    <property>
        <name>dfs.ha.automatic-failover.enabled</name>
        <value>true</value>
    </property>
    <property>
        <name>dfs.client.block.write.replace-datanode-on-failure.policy</name>
        <value>NEVER</value>
    </property>
</configuration>

```

- mapred-site.xml

```xml
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>

```

- yarn-site.xml

```xml
<?xml version="1.0"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.resourcemanager.ha.enabled</name>
        <value>true</value>
    </property>
    <property>
        <name>yarn.resourcemanager.cluster-id</name>
        <value>rmCluster</value>
    </property>
    <property>
        <name>yarn.resourcemanager.ha.rm-ids</name>
        <value>rm1,rm2</value>
    </property>
    <property>
        <name>yarn.resourcemanager.webapp.address</name>
        <value>192.168.0.1:8088</value>
    </property>
    <property>
        <name>yarn.resourcemanager.hostname.rm1</name>
        <value>192.168.0.1</value>
    </property>
    <property>
        <name>yarn.resourcemanager.hostname.rm2</name>
        <value>192.168.0.2</value>
    </property>
    <property>
        <name>yarn.resourcemanager.zk-address</name>
        <value>192.168.0.1:2181,192.168.0.2:2181,192.168.0.3:2181</value>
    </property>
    <property>
        <name>yarn.resourcemanager.recovery.enabled</name>
        <value>true</value>
    </property>
    <property>
        <name>yarn.resourcemanager.store.class</name>
        <value>org.apache.hadoop.yarn.server.resourcemanager.recovery.ZKRMStateStore</value>
    </property>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
        <value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
    <property>
        <name>yarn.nodemanager.pmem-check-enabled</name>
        <value>false</value>
    </property>

    <property>
        <name>yarn.nodemanager.vmem-check-enabled</name>
        <value>false</value>
    </property>

    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>20480</value>
    </property>
    <property>
        <name>yarn.nodemanager.disk-health-checker.max-disk-utilization-per-disk-percentage</name>
        <value>97.0</value>
    </property>
</configuration>

```

**\#Make New Directory**

```bash
cd  /data/projects/common/hadoop
mkdir ./tmp
mkdir -p ./data/dfs/nn/local
```

**\#Start**

Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:

```bash
hadoop-daemon.sh start journalnode
```

Execute on 192.168.0.1 as user app:

```bash
hdfs namenode -format
hadoop-daemon.sh start namenode
```

Execute on 192.168.0.2 as user app:

```bash
hdfs namenode -bootstrapStandby
```

Execute on 192.168.0.1 as user app:

```bash
hdfs zkfc -formatZK
```

Execute on 192.168.0.2 as user app:

```bash
hadoop-daemon.sh start namenode
```

Execute on 192.168.0.2 as user app:

```bash
hadoop-daemon.sh start zkfc
```

Execute on 192.168.0.2 as user app:

```bash
yarn-daemon.sh start resourcemanager
```

Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:

```bash
yarn-daemon.sh start nodemanager
```

Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:

```bash
hadoop-daemon.sh start datanode
```

**\#Check**

Check Hadoop status at http://192.168.0.1:50070

Check yarn cluster status at http://192.168.0.1:8088

## 6.Spark Cluster Deploy

**\#Execute on 192.168.0.1 192.168.0.2 192.168.0.3 as user app:**

```bash
cd /data/projects/common/spark/conf
cat slaves
```

Add: VM-0-2-centos VM-0-3-centos

```bash
cat spark-defaults.conf
```

Add:

spark.master yarn

spark.eventLog.enabled true

spark.eventLog.dir hdfs://fate-cluster/tmp/spark/event

\# spark.serializer org.apache.spark.serializer.KryoSerializer

\# spark.driver.memory 5g

\# spark.executor.extraJavaOptions -XX:+PrintGCDetails -Dkey=value
-Dnumbers="one two three"

spark.yarn.jars hdfs://fate-cluster/tmp/spark/jars/\*.jar

**Add to spark-env.sh: **

```
export JAVA_HOME=/data/projects/common/jdk/jdk-8u192
export SCALA_HOME=/data/projects/common/scala
export HADOOP_HOME=/data/projects/common/hadoop
export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop
export
SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://fate-cluster/tmp/spark/event"
export HADOOP_COMMON_LIB_NATIVE_DIR=\$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=\$HADOOP_HOME/lib/native"
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${HADOOP_HOME}/lib/native
export PYSPARK_PYTHON=/data/projects/fate/common/python/venv/bin/python
export PYSPARK_DRIVER_PYTHON=/data/projects/fate/common/python/venv/bin/python
```

**\#Start**

```bash
sh /data/projects/common/spark/sbin/start-all.sh
```

**\#Check**

```bash
cd /data/projects/common/spark/jars
hdfs dfs -mkdir -p /tmp/spark/jars
hdfs dfs -mkdir -p /tmp/spark/event
hdfs dfs -put *jar /tmp/spark/jars
/data/projects/common/spark/bin/spark-shell --master yarn --deploy-mode client 
```
