
# eggroll升级工具文档说明
此文档兼容eggroll2.0.x -> 2.2.1

## 1. 环境要求
##### 1.1   python3环境(正常安装的官方python3即可，无第三方依赖)
##### 1.2  执行脚本的机器需要能免密登录app用户Eggroll集群的所有节点
##### 1.3  执行升级脚本的机器需要app用户登录执行（如果无法使用app用户，那么需要保证运行升级脚本的用户可以免密app用户下的eggroll集群所有节点）
##### 1.4  执行mysql备份等操作需要先确认是否允许ip操作权限

## 2. 参数说明
```
 -c --nm_file <必选:eggroll集群cm或nm节点ip集合>
 -r --rs_file <必选:eggroll集群仅仅部署rollsite节点的ip集合,内容可以为空,文件不能为空>
 -e --egg_home <必选:eggroll home目录>
 -k --pkg_base_path <必选:eggroll 升级包的父目录路径，详见3.1.1>
 -m --mysql_home <必选:mysql home 目录>
 -t --mysql_host  <必选:mysql主机ip地址>
 -p --mysql_port <必选:mysql端口号>
 -b --mysql_db <必选:mysql eggroll元数据库名称>
 -u --mysql_user <必选:mysql  eggroll元数据库登录用户名称>
 -w --mysql_pwd <必选:mysql eggroll元数据库登录用户密码>
 -S --mysql-sock <必选:mysql sock文件路径>
 -s --mysql_file <必选:mysql eggroll元数据修改内容sql文件集>
 -f --recover <可选项:eggroll 升级失败恢复默认参数0升级,1 回滚>
```

## 升级前准备

- 切换虚拟环境
  进入fate安装目录,找到环境变量shell

  如果fate版本为1.4.x：

  ```
  eg:
  source /data/projects/fate/init_env.sh
  ```

  如果fate版本为1.5.x

  ```
  eg:
  source /data/projects/fate/bin/init_env.sh
  ```
- 停服务

> 进入eggroll home目录

- allinone 部署方式的停止服务方法
```
cd ${EGGROLL_HOME}
sh bin/eggroll.sh all stop
```
- ansible 部署方式的停止服务方法
```
cd /data/projects/common/supervisord
sh service.sh stop fate-clustermanager
sh service.sh stop fate-nodemanager
sh service.sh stop fate-rollsite
```

- eggroll手动备份

只需要在脚本执行所在机器节点执行一次
```
cd ${EGGROLL_HOME}
cp -r bin bin_bak
cp -r deploy deploy_bak
cp -r lib lib_bak
cp -r python python_bak
```

- 手动备份eggroll元数据库

命令行登录mysql,切换eggroll_meta（当前eggroll集群的元数据库名）

> 1、只需备份`eggroll_meta`数据库即可

> 2、${MYSQL_HOME_PATH}为MYSQL安装路径

> 3、注意-p后面不能留空格

```
${MYSQL_HOME_PATH}/bin/mysqldump -h <mysql-host> -u <username> -p<passwd> -P <port> -S <sock-file> eggroll_meta > eggroll_meta_dump_bak.sql
```



## 3. 使用说明

### 3.1 

#### 3.1.1 升级脚本部署

##### 3.1.1.2 获取升级包

- a. 若升级包中提供了编译包: eggroll.tar.gz，则直接解压使用即可
- b. 若需要进行源码编译，则
  - 从https://github.com/WeBankFinTech/eggroll/blob/dev-2.2.1获取源码，并且编译jar包后将jar包放入lib目录下，详细操作可参考：
    https://github.com/WeBankFinTech/eggroll/blob/main/deploy/Eggroll%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3%E8%AF%B4%E6%98%8E.md

> 升级包目录结构如下:

```
├─eggroll
      ├─bin 
      ├─deploy 
      ├─lib  
      └─python 
   
```

##### 3.1.1.1 获取升级脚本

```
source ${PYTHON3_VENV_PATH}
export PKG_BASE_PATH={You upgrade the package one level above the eggroll directory}
python ${PKG_BASE_PATH}/eggroll/deploy/upgrade_helper.py --help
```

##### 3.1.1.3 写入版本号(在eggroll集群所有节点机器操作)

###### a.如果从`eggroll_2.0.1 -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)`

进入`/data/projects/fate/eggroll`目录，执行以下操作

```
echo "__version__ = "2.0.1"" > /data/projects/fate/eggroll/python/eggroll/__init__.py
```

###### b. 如果`eggroll_2.0.2 -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)`

`/data/projects/fate/eggroll`目录，执行以下操作

```
echo "__version__ = "2.0.2"" > /data/projects/fate/eggroll/python/eggroll/__init__.py
```



##### 3.1.1.4 修改默认配置

***该步骤需要在Eggroll集群的所有节点上执行，并且需要重启生效***

```
cd /data/projects/fate/eggroll/conf
vim eggroll.properties
增加以下参数：
eggroll.core.grpc.channel.keepalive.timeout.sec=20

修改以下参数
eggroll.rollsite.adapter.sendbuf.size=100000

保存后退出vim
```



##### 3.1.1.4 部署镜像工程

***如果直接在Eggroll集群运行升级脚本，则可跳过该步骤***

***该步骤仅针对非Eggroll集群机器（如跳板机）升级指定Eggroll集群所用***

执行升级脚本的机器需要与Eggroll集群有相同的目录结构，例如：

当前有执行升级脚本的机器A（机器A上未部署Eggroll集群），升级目标机器B上的eggroll home为: `/data/projects/fate/eggroll` ,那么机器A需要部署一个与机器B相同的目录，操作如下：

###### a.如果从`eggroll_2.0.1 -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)`

那么需要从https://github.com/WeBankFinTech/eggroll/releases/tag/v2.0.1获取源码包，上传并解压至机器A的

`/data/projects/fate/`目录，执行以下操作

```
mkdir -p /data/projects/fate/ 
unzip /data/projects/fate/eggroll-2.0.1.zip
cp -r /data/projects/fate/eggroll-2.0.1 /data/projects/fate/eggroll
echo "__version__ = "2.0.1"" > /data/projects/fate/eggroll/python/eggroll/__init__.py
```

###### b. 如果`eggroll_2.0.2 -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)`

那么需要从https://github.com/WeBankFinTech/eggroll/releases/tag/v2.0.2获取源码包，上传并解压至机器A的

`/data/projects/fate/`目录，执行以下操作

```
mkdir -p /data/projects/fate/ 
unzip /data/projects/fate/eggroll-2.0.2.zip
cp -r /data/projects/fate/eggroll-2.0.2 /data/projects/fate/eggroll
echo "__version__ = "2.0.2"" > /data/projects/fate/eggroll/python/eggroll/__init__.py
```

###### c. 如果`eggroll_2.2.0 -> 2.2.1（对应FATE 1.4.x -> FATE 1.5.1)`

那么需要从https://github.com/WeBankFinTech/eggroll/releases/tag/v2.2.0获取源码包，上传并解压至机器A的

`/data/projects/fate/`目录，执行以下操作

```
mkdir -p /data/projects/fate/ 
unzip /data/projects/fate/eggroll-2.2.0.zip
cp -r /data/projects/fate/eggroll-2.2.0 /data/projects/fate/eggroll
```

#### 3.1.2 创建nm_ip_list文件

即需要升级eggroll版本所在的nodemanager节点ip列表集合

```

touch nm_ip_list
vim  nm_ip_list
192.168.0.1
192.168.0.2

```

### 3.1.3 rs_ip_list文件

即需要升级eggroll版本所在的rollsite节点ip列表集合,可以为空,文件不能不存在
此文件仅仅加入独立部署的rollsite节点ip,如果nodemanager与rollsite部署在同一个机器上则无需配置该文件

```
touch  rs_ip_list
```

### 3.1.4 mysql_file.sql文件

即升级eggroll元数据修改内容sql文件集
此文件不能包含带有注释的sql语句,否则升级失败

- 此文件需要根据eggroll升级版本号变更eggroll元数据库

> 1、eggroll_2.0.x -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)

```
touch mysql_file.sql
vim mysql_file.sql

use eggroll_meta;
alter table store_option modify column store_locator_id bigint unsigned;
alter table store_option add store_option_id SERIAL PRIMARY KEY;
alter table session_option add store_option_id SERIAL PRIMARY KEY;
alter table session_main modify column session_id VARCHAR(767);
alter table session_processor modify column session_id VARCHAR(767);
```

> 2、eggroll_2.2.x -> eggroll_2.2.x（对应FATE 1.5.0 -> FATE 1.5.1)

```
touch mysql_file.sql
vim mysql_file.sql

use eggroll_meta;
alter table store_option add store_option_id SERIAL PRIMARY KEY;
alter table session_option add store_option_id SERIAL PRIMARY KEY;
alter table session_main modify column session_id VARCHAR(767);
alter table session_processor modify column session_id VARCHAR(767);


```

## 4 脚本执行

- 4.1 使用-h 打印命令行帮助

```
python ${PKG_BASE_PATH}/eggroll/deploy/upgrade_helper.py --help
python upgrade_helper.py 
 -c --nm_file <input eggroll upgrade  namenode node ip sets>
 -r --rs_file <input eggroll upgrade only rollsite node ip sets>
 -e --egg_home <eggroll home path>
 -k --pkg_base_path <The base path of the EggRoll upgrade package,This directory contains the following files:见3.1.1>
 -m --mysql_home <mysql home path>
 -t --mysql_host <mysql ip addr>
 -p --mysql_port <mysql port>
 -b --mysql_db <mysql database>
 -u --mysql_user <mysql username>
 -w --mysql_pwd <mysql passwd>
 -S --mysql-sock <mysql sock文件路径>
 -s --mysql_file <mysql upgrade content sql file sets>
 -f --recover <upgrade fail recover default 0 upgrade recover 1 recover>
```

- 4.2 启动升级

```
python ${PKG_BASE_PATH}/eggroll/deploy/upgrade_helper.py \
-c ${your create nm_ip_list file path contains the file name} \
-r ${your create rs_ip_list file path contains the file name} \
-e ${EGGROLL_HOME} \
-k ${PKG_BASE_PATH} \
-m ${mysql install home path} \
-t ${your install fate with mysql ip} \
-p ${your install fate with mysql port} \
-b eggroll_meta \
-u ${your install fate with mysql username} \
-w ${your install fate with mysql passwd} \
-S ${MYSQL_HOME}/run/mysql.sock \
-s ${your create mysql_file sql file path contains the file name} \
> log.info

```
> 执行一次失败后,恢复`eggroll手动备份`方可再次执行升级脚本

> 以上文件、目录均为绝对路径

- 4.3 检查日志
```
less log.info | grep "current machine not install mysql "
```

如果当前日志中出现了该日志，则代表当前升级脚本所在机器没有安装MYSQL

> 1、前往mysql所在机器

> 2、登录

> 3、执行以下操作：
>
> > 1、eggroll_2.0.x -> 2.2.x（对应FATE 1.4.x -> FATE 1.5.1)
>
> ```
> use eggroll_meta;
> alter table store_option modify column store_locator_id bigint unsigned;
> alter table store_option add store_option_id SERIAL PRIMARY KEY;
> alter table session_option add store_option_id SERIAL PRIMARY KEY;
> alter table session_main modify column session_id VARCHAR(767);
> alter table session_processor modify column session_id VARCHAR(767);
> ```
>
> > 2、eggroll_2.2.x -> eggroll_2.2.x（对应FATE 1.5.0 -> FATE 1.5.1)
>
> ```
> use eggroll_meta;
> alter table store_option add store_option_id SERIAL PRIMARY KEY;
> alter table session_option add store_option_id SERIAL PRIMARY KEY;
> alter table session_main modify column session_id VARCHAR(767);
> alter table session_processor modify column session_id VARCHAR(767);
> 
> 
> ```
>
> ## 


- 4.4 版本检查
```
cat $EGGROLL_HOME/python/eggroll/__init__.py

```


## 5. 升级失败恢复EGGROLL集群所有的升级节点

- 5.1 eggroll还原

进入升级脚本所在节点的eggroll_home目录 -> 将bak结尾的目录或文件删除新的复制一份备份目录

```
cd ${EGGROLL_HOME}
rm -rf bin deploy lib python
cp -r bin_bak bin
cp -r deploy_bak deploy
cp -r lib_bak lib
cp -r python_bak python

```

- 5.2 EGGROLL元数据库恢复

```
${MYSQL_HOME}/bin/mysql -ufate -p -S ${MYSQL_HOME}/run/mysql.sock -h 192.168.0.1 -P 3306 --default-character-set=utf8 eggroll_meta < dump_bakbak.sql
```
- 5.3 回滚执行

```
python ${PKG_BASE_PATH}/deploy/upgrade_helper.py \
-c ${your create nm_ip_list file path contains the file name} \
-r ${your create rs_ip_list file path contains the file name} \
-e ${EGGROLL_HOME} \
-k ${PKG_BASE_PATH} \
-m ${MYSQL_HOME} \
-t ${your install fate with mysql ip} \
-p ${your install fate with mysql port} \
-b eggroll_meta \
-u ${your install fate with mysql username} \
-w ${your install fate with mysql passwd} \
-S ${MYSQL_HOME}/run/mysql.sock \
-s ${your create mysql_file sql file path contains the file name} \
-f 1
> recover.info
```


- 5.4 重复#4.3 ~ 4.4步骤

## 6. 常见问题与注意事项

- 6.1

> **Q:** 如果以前ssh不能直接登录到目标机器，需要指定端口，怎么办呢？

> **A:** 在执行升级脚本前，执行以下语句，指定ssh端口：


```
export "RSYNC_RSH=ssh -p ${ssh_port}"
echo $RSYNC_RSH
```


> 其中${ssh_port}为以前登录到目标机器时需要指定的端口。


- 6.2


> **Q:** 如果我使用ansible模式部署fate，有什么要注意呢?

> **A:** 使用ansible部署fate，需要在完成升级脚本后进行以下操作：

```
vim ${EGGROLL_HOME}/bin/eggroll.sh
vim内搜索：{module}.err &
定位到改行，删除行末的：&
保存后退出vim
cp ${EGGROLL_HOME}/bin/eggroll.sh ${EGGROLL_HOME}/bin/fate-eggroll.sh
sed -i '26 i source $cwd/../../bin/init_env.sh' ${EGGROLL_HOME}/bin/fate-eggroll.sh
```

