# Source Code Deployment FATE Standalone

[TOC]

## 1. Description

Server configuration.

| **number**        | 1                                    |
| ----------------- | ------------------------------------ |
| **Configuration** | 8 core / 16G memory / 500G hard disk |
| **OS**            | Version: CentOS Linux release 7      |
| **User**          | User: app owner:apps                 |

## 2. Pre-deployment environment check

Whether local ports 8080, 9360, 9380 are occupied

```bash
netstat -apln|grep 8080
netstat -apln|grep 9360
netstat -apln|grep 9380
```

## 3. Get the source code

Please refer to [get source code](../../../build/common/get_source_code.md), and then,

Set the environment variables required for deployment (note that the environment variables set in the following way are only valid for the current terminal session, if you open a new terminal session, such as a new login or a new window, please set them again)

```bash
cd {The directory where the above code is stored}
export FATE_PROJECT_BASE=$PWD
export version=`grep "FATE=" ${FATE_PROJECT_BASE}/fate.env | awk -F "=" '{print $2}'`
```

example:

```bash
cd /xxx/FATE
export FATE_PROJECT_BASE=$PWD
export version=`grep "FATE=" ${FATE_PROJECT_BASE}/fate.env | awk -F "=" '{print $2}'`
```

## 4. Install and configure the Python environment

### 4.1 Installing the Python environment (optional)

Please install or use existing Python version 3.8

### 4.2 Configuring a virtual environment for FATE

```bash
cd(or create) {root directory for the virtual environment}
python3 -m venv {虚拟环境名称}
export FATE_VENV_BASE={root directory for the virtual environment}/{name of virtual environment}
source ${FATE_VENV_BASE}/bin/activate
```

### 4.3 Installing Python dependencies for FATE

```bash
cd ${FATE_PROJECT_BASE};
bash bin/install_os_dependencies.sh;
source ${FATE_VENV_BASE}/bin/activate;
pip install -r python/requirements.txt
```

In case of problems, you can first refer to [Possible problems](#11-problems-that-may-be-encountered)

## 5. Configuring FATE

Edit the `bin/init_env.sh` environment variable file

```bash
cd ${FATE_PROJECT_BASE};
sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=$PWD/python:$PWD/fateflow/python#g" bin/init_env.sh;
sed -i.bak "s#venv=.*#venv=${FATE_VENV_BASE}#g" bin/init_env.sh
```

Check if the `conf/service_conf.yaml` global configuration file has the base engine configured as standalone, if `default_engines` shows the following, then it is standalone

```yaml
default_engines:
  computing: standalone
  federation: standalone
  storage: standalone
```

## 6. start fate flow server

```bash
cd ${FATE_PROJECT_BASE};
source bin/init_env.sh;
cd fateflow;
bash bin/service.sh status;
bash bin/service.sh start
```

If it shows something like this, it is started successfully, otherwise please check the logs according to the prompt

```bash
service start sucessfully. pid: 111907
status:app 111907 75.7 1.1 3740008 373448 pts/2 Sl+ 12:21 0:17 python /xx/FATE/fateflow/python/fate_flow/fate_flow_server.py
python 111907 app 14u IPv4 3570158828 0t0 TCP localhost:boxp (LISTEN)
python 111907 app 13u IPv4 3570158827 0t0 TCP localhost:9360 (LISTEN)
```

## 7. install fate client

```bash
cd ${FATE_PROJECT_BASE};
source bin/init_env.sh;
cd python/fate_client/;
python setup.py install
```

Initialize ``fate flow client`''

```bash
cd ../../;
flow init -c conf/service_conf.yaml
```

If it looks like this, the initialization is successful, otherwise, please check the logs according to the prompt

```json
{
    "retcode": 0,
    "retmsg": "Fate Flow CLI has been initialized successfully."
}
```

## 8. Test items

### 8.1 Toy test

   ```bash
   flow test toy -gid 10000 -hid 10000
   ```

   If successful, the screen displays a statement similar to the following:

   ```bash
   success to calculate secure_sum, it is 2000.0
   ```

### 8.2 Unit tests

   ```bash
   cd ${FATE_PROJECT_BASE};
   bash python/federatedml/test/run_test.sh
   ```

   If successful, the screen displays a statement similar to the following:

   ```bash
   there are 0 failed test
   ```

Some use case algorithms are in [examples](../../../examples/dsl/v2) folder, please try to use them.
Please refer to [here](../../../examples/pipeline/../README.md) for a quick start tutorial.

You can also experience the algorithm process kanban through your browser, please refer [here](#9-compile-package-to-install-fateboard-optional)

## 9. Compile package to install fateboard (optional)

Visualizing FATE Jobs with fateboard

### 9.1 Installing and configuring the Java environment

```bash
cd ${FATE_PROJECT_BASE};
mkdir -p env/jdk;
cd env/jdk;
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/jdk-8u192.tar.gz;
tar xzf jdk-8u192.tar.gz
```

Configure environment variables

```bash
cd ${FATE_PROJECT_BASE};
vim bin/init_env.sh;
sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=$PWD/env/jdk/jdk-8u192/#g" bin/init_env.sh
```

### 9.2 Download the build package to install fateboard

```bash
cd ${FATE_PROJECT_BASE};
mv fateboard fateboard_code;
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/fateboard.tar.gz;
tar xzf fateboard.tar.gz;
sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:$PWD/fate_sqlite.db#g" $PWD/fateboard/conf/application.properties;
sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" $PWD/fateboard/conf/application.properties
```

### 9.3 Starting fateboard

```bash
cd fateboard;
bash service.sh status;
bash service.sh start
```

If you see something like the following, you have started successfully, otherwise please check the logs as prompted

```bash
JAVA_HOME=/data/project/deploy/FATE/env/jdk/jdk-8u192/
service start sucessfully. pid: 116985
status:
        app 116985 333 1.7 5087004 581460 pts/2 Sl+ 14:11 0:06 /xx/FATE/env/jdk/jdk-8u192//bin/java -Dspring.config.location=/xx/FATE/fateboard/conf/ application.properties -Dssh_config_file=/xx/FATE/fateboard/ssh/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc: gc.log -XX:+HeapDumpOnOutOfMemoryError -jar /xx/FATE/fateboard
```

## 10. Source code installation of fateboard

Please refer to the [FATEBoard repository](https://github.com/FederatedAI/FATE-Board)

## 11. Problems that may be encountered

- If you get an error like "Too many open files", it may be because the number of OS handles is too low.
  - For MacOS, you can try [here](https://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1 )
  - For Linux, you can try [here](http://woshub.com/too-many-open-files-error-linux/)

- If the installation of the `python` dependency package `gmpy2` fails under MacOS, try installing the following base library before installing the dependency package

```bash
brew install gmp mpfr libmpc
```
