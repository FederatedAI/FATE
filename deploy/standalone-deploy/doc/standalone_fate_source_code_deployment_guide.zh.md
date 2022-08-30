# 源码部署FATE单机版

[TOC]

## 1. 说明

服务器配置：

| **数量**     | 1                                    |
| ------------ | ------------------------------------ |
| **配置**     | 8 core / 16G memory / 500G hard disk |
| **操作系统** | Version: CentOS Linux release 7      |
| **用户**     | User: app owner:apps                 |

注意，如下示例中的${version}，请用实际的版本号替换，参考[fate.env](../../../fate.env)文件中的FATE版本

## 2. 部署前环境检查

本地8080、9360、9380端口是否被占用

```bash
netstat -apln|grep 8080
netstat -apln|grep 9360
netstat -apln|grep 9380
```

## 3. 获取源代码

请参考[获取源代码](../../../build/common/get_source_code.zh.md), 完成后,

请设置部署所需环境变量(注意, 通过以下方式设置的环境变量仅在当前终端会话有效, 若打开新的终端会话, 如重新登录或者新窗口, 请重新设置)

```bash
cd {上述代码的存放目录}
export FATE_PROJECT_BASE=$PWD
export version=`grep "FATE=" ${FATE_PROJECT_BASE}/fate.env | awk -F "=" '{print $2}'`
```

样例:

```bash
cd /xxx/FATE
export FATE_PROJECT_BASE=$PWD
export version=`grep "FATE=" ${FATE_PROJECT_BASE}/fate.env | awk -F "=" '{print $2}'`
```

## 4. 安装并配置Python环境

### 4.1 安装Python环境(可选)

请安装或者使用已有的 Python 3.8 版本

### 4.2 为FATE配置虚拟环境

```bash
cd(or create) {放置虚拟环境的根目录}
python3 -m venv {虚拟环境名称}
export FATE_VENV_BASE={放置虚拟环境的根目录}/{虚拟环境名称}
source ${FATE_VENV_BASE}/bin/activate
```

### 4.3 安装FATE所需要的Python依赖包

```bash
cd ${FATE_PROJECT_BASE};
bash bin/install_os_dependencies.sh;
source ${FATE_VENV_BASE}/bin/activate;
pip install -r python/requirements.txt
```

如果出现相关问题, 可以先参考[可能遇到的问题](#11-可能会遇到的问题)

## 5. 配置FATE

编辑`bin/init_env.sh`环境变量文件

```bash
cd ${FATE_PROJECT_BASE}
sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=$PWD/python:$PWD/fateflow/python#g" bin/init_env.sh;
sed -i.bak "s#venv=.*#venv=${FATE_VENV_BASE}#g" bin/init_env.sh
```

检查`conf/service_conf.yaml`全局配置文件中是否将基础引擎配置为单机版, 若`default_engines`显示如下，则为单机版

```yaml
default_engines:
  computing: standalone
  federation: standalone
  storage: standalone
```

## 6. 启动fate flow server

```bash
cd ${FATE_PROJECT_BASE};
source bin/init_env.sh;
cd fateflow;
bash bin/service.sh status;
bash bin/service.sh start
```

显示如下类似则为启动成功，否则请依据提示查看日志

```bash
service start sucessfully. pid: 111907
status:app      111907 75.7  1.1 3740008 373448 pts/2  Sl+  12:21   0:17 python /xx/FATE/fateflow/python/fate_flow/fate_flow_server.py
python  111907  app   14u  IPv4 3570158828      0t0  TCP localhost:boxp (LISTEN)
python  111907  app   13u  IPv4 3570158827      0t0  TCP localhost:9360 (LISTEN)
```

## 7. 安装fate client

```bash
cd ${FATE_PROJECT_BASE};
source bin/init_env.sh;
cd python/fate_client/;
python setup.py install
```

初始化`fate flow client`

```bash
cd ../../;
flow init -c conf/service_conf.yaml
```

显示如下类似则为初始化成功，否则请依据提示查看日志

```json
{
    "retcode": 0,
    "retmsg": "Fate Flow CLI has been initialized successfully."
}
```

## 8. 测试项

### 8.1 Toy测试

   ```bash
   flow test toy -gid 10000 -hid 10000
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   success to calculate secure_sum, it is 2000.0
   ```

### 8.2 单元测试

   ```bash
   cd ${FATE_PROJECT_BASE};
   bash ./python/federatedml/test/run_test.sh
   ```

   如果成功，屏幕显示类似下方的语句:

   ```bash
   there are 0 failed test
   ```

有些用例算法在 [examples](../../../examples/dsl/v2) 文件夹下, 请尝试使用。
Please refer [here](../../../examples/pipeline/../README.zh.md) for a quick start tutorial.

您还可以通过浏览器体验算法过程看板，请参照[编译包安装fateboard](#9-编译包安装fateboard建议可选)

## 9. 编译包安装fateboard(建议可选)

使用fateboard可视化FATE Job

### 9.1 安装并配置Java环境

```bash
cd ${FATE_PROJECT_BASE};
mkdir -p env/jdk;
cd env/jdk
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/jdk-8u192.tar.gz;
tar xzf jdk-8u192.tar.gz
```

配置环境变量

```bash
cd ${FATE_PROJECT_BASE};
vim bin/init_env.sh;
sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=$PWD/env/jdk/jdk-8u192/#g" bin/init_env.sh
```

### 9.2 下载编译包安装fateboard

```bash
cd ${FATE_PROJECT_BASE};
mv fateboard fateboard_code;
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/fateboard.tar.gz;
tar xzf fateboard.tar.gz;
sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:$PWD/fate_sqlite.db#g" $PWD/fateboard/conf/application.properties;
sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" $PWD/fateboard/conf/application.properties
```

### 9.3 启动fateboard

```bash
cd fateboard;
bash service.sh status;
bash service.sh start
```

显示如下类似则为启动成功，否则请依据提示查看日志

```bash
JAVA_HOME=/data/project/deploy/FATE/env/jdk/jdk-8u192/
service start sucessfully. pid: 116985
status:
        app      116985  333  1.7 5087004 581460 pts/2  Sl+  14:11   0:06 /xx/FATE/env/jdk/jdk-8u192//bin/java -Dspring.config.location=/xx/FATE/fateboard/conf/application.properties -Dssh_config_file=/xx/FATE/fateboard/ssh/ -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError -jar /xx/FATE/fateboard/fateboard.jar
```

访问：Http://${ip}:8080, ip为`127.0.0.1`或本机实际ip

## 10. 源码安装fateboard

请参考[FATEBoard仓库](https://github.com/FederatedAI/FATE-Board)

## 11. 可能会遇到的问题

- 如果出现"Too many open files"类似错误，可能是因为操作系统句柄数配置过低
  - 对于MacOS, 可以尝试[这里](https://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1)
  - 对于Linux, 可以尝试[这儿](http://woshub.com/too-many-open-files-error-linux/)

- 如果在MacOS下面, 安装`gmpy2`这个`python`依赖包失败的话, 尝试先安装如下基础库后, 再安装依赖包

```bash
brew install gmp mpfr libmpc
```
