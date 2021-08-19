# FATE v1.5.0 - v1.5.1升级包使用说明

**本使用说明仅适用于FATE v1.5.0 - v1.5.1的更新，说明中更新包下载路径及解压目录均以用户的home目录为例。**


## 1. 检查升级包

应该包括`upgrade.sh`脚本及若干升级所用目录


## 2. 参数修改

进入升级包的解压路径，对解压路径下的`upgrade.sh`脚本进行修改，需要修改的参数位于该脚本的5-14行中，以下是对这些参数的解释说明：

```bash
# FATE项目根目录
FATE_ROOT=/data/projects/fate

# MySQL高权账号（需提供高权账号且通过socket进行数据库连接，否则无法成功更新数据库）
DB_USER=root

# MySQL高权账号的密码
DB_PASS=fate_dev

# MySQL根目录
MYSQL_ROOT=$FATE_ROOT/common/mysql/mysql-8.0.13

# MySQL Sock文件路径
MYSQL_SOCKET_PATH=$MYSQL_ROOT/run/mysql.sock

# 原部署方式（需从allinone和ansible中进行选择修改）
DEPLOY_METHOD=allinone

# 如果部署方式为ansible的情况下，还需指定ansible安装的supervisor根目录
SUPERVISOR_ROOT=/data/projects/common/supervisord
```



## 3. 执行更新脚本

对参数进行修改后，请对更新脚本进行保存。保存后，将升级包目录拷贝到需要升级的机器，使用如下命令执行更新：

请使用如下命令执行更新：
### 3.1 更新脚本使用方法说明

```bash
cd ~/upgrade_1_5_0-1_5_1/

# 带参数执行如下命令，其中组件名请从：fatepython、fateflow、mysql、all中选择
sh upgrade.sh <module>

# 例如，如果只需要更新fatepython，请执行如下命令：
sh upgrade.sh fatepython

# 若所有服务均部署在同一台机器，请使用all，否则不能使用all:
sh upgrade.sh all
```

如果提示ERROR，... aborting字样，则为参数检查不通过，请根据提示对脚本参数进行二次确认及修改；

如果提示Upgrading process finished字样，则更新成功。

**【注意】**如果选择单组件分别更新，更新数据库时请手动停止fateflow服务，否则数据库更新进程可能会失败。手动停止fateflow服务的方法详见4.1.1。

### 3.2 更新mysql(mysql所在机器)

```bash
sh upgrade.sh mysql
```

### 3.3 更新fatepython(fateflow、nodemanager所在机器)

```bash
sh upgrade.sh fatepython
```

### 3.4 更新fateboard(fateboard所在机器)

```bash
sh upgrade.sh fateboard
```

## 4. 更新回滚

### 4.1 更新所有组件后的回滚操作

#### 4.1.1 停止fate_flow_server服务

##### allinone部署方式的停止服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 停止服务
sh service.sh stop
```

##### ansible部署方式的停止服务方法
```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 停止服务
sh service.sh stop fate-fateflow
```



#### 4.1.2 代码更新回退

FATE python代码包在更新时会在同级目录下备份，文件夹名为`python_150bak_更新时间`，例如`python_150bak_202101011200`。如需回退，请将更新后的python代码包删除或进行备份，并将原python代码包的名字重新修改为`python`即可。具体操作如下：

```bash
# 进入FATE项目根目录
cd /data/projects/fate/

# 将更新后的python代码包进行备份
mv python/ python_upgrade_backup/

# 将原python代码包重命名为python
mv python_150bak_202101011200/ python/
```



#### 4.1.3 数据库更新回退

```mysql
# 使用高权账号登录MySQL，登录后执行如下操作
# 如果用户使用FATE默认路径安装MySQL，则可以通过如下命令连入数据库：
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock

# 选择fate_flow数据库
use fate_flow

# 对更新过的数据表进行备份
alter table t_machine_learning_model_info rename to t_queue_backup_20201201;
alter table t_job rename to t_job_backup_20201201;

# 将原数据库表进行恢复
alter table t_machine_learning_model_info_backup150 rename to t_machine_learning_model_info;
alter table t_job_backup150 rename to t_job;
```



#### 4.1.4 启动fate_flow_server服务

##### allinone部署方式的启动服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 启动服务
sh service.sh start
```

##### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 启动服务
sh service.sh start fate-fateflow
```



#### 4.1.5 停止fateboard服务

##### allinone部署方式的停止服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/fateboard/

# 停止服务
sh service.sh stop
```

##### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 停止服务
sh service.sh stop fate-fateboard
```



#### 4.1.6 fateboard更新回退

FATEBOARD在更新时会在同级目录下备份，文件夹名为`fateboard_150bak_更新时间`，例如`fateboard_150bak_202101011200`。如需回退，请将更新后的fateboard目录删除或进行备份，并将原fateboard目录的名字重新修改为`fateboard`即可。具体操作如下：

```bash
# 进入FATE项目根目录
cd /data/projects/fate/

# 将更新后的fateboard代码包进行备份
mv fateboard/ fateboard_upgrade_backup/

# 将原fateboard代码包重命名为python
mv fateboard_150bak_202101011200/ fateboard/
```



#### 4.1.7 启动fateboard服务

##### allinone部署方式的停止服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/fateboard/

# 停止服务
sh service.sh start
```

##### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 停止服务
sh service.sh start fate-fateboard
```



### 4.2 仅更新fatepython组件后的回滚操作

详细操作请见4.1.1、4.1.2、4.1.4步骤。



### 4.3 仅更新mysql组件后的回滚操作

详细操作请见4.1.3步骤。



### 4.4 仅更新fateboard组件后的回滚操作

详细操作请见4.1.5、4.1.6、4.1.7步骤。