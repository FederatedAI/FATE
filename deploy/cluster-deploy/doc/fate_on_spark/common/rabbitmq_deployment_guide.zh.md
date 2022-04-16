### 一. 环境准备

1. rabbitmq-server-generic-unix-3.6.15.tar.xz
2. otp_src_19.3.tar.gz

### 二. 安装Erlang

1. 下载Erlang源码(otp_src_19.3.tar.gz)，并解压至/data/projects/fate/common

   ```bash
   tar -zxvf otp_src_19.3.tar.gz -C /data/projects/fate/common
   ```

2. 配置ERL_TOP

   ```bash
   cd  /data/projects/fate/common/otp_src_19.3/
   export ERL_TOP=`pwd`
   ```

3. 编译

   使用以下命令编译:

   ```bash
   ./configure --prefix=/data/projects/fate/common/erlang
   make
   make install
   ```

   如果出现 **No curses library functions found**报错，则需要安装ncuress，先下载ncurses-6.0.tar.gz

   ```bash
   tar -zxvf ncurses-6.0.tar.gz
   cd ncurses-6.0
   ./configure --with-shared --without-debug --without-ada --enable-overwrite  
   make
   make install (如果报Permission denied，则需要root权限执行)
   ```

4. 设置环境变量

   编译完成后，设置 ERL_HOME。编辑 /etc/profile 文件，增加以下内容：

   ```bash
   cat >> /etc/profile << EOF
   export ERL_HOME=/data/projects/fate/common/erlang
   export PATH=$PATH:/data/projects/fate/common/erlang/bin
   EOF
   ```

5.  验证

   执行命令: erl, 可以进入Erlang环境，则安装成功；

### 三. 安装RabbitMQ

1. **下载RabbitMq Server安装包，并解压至/data/projects/fate/common**

   ```bash
   xz -d rabbitmq-server-generic-unix-3.6.15.tar.xz
   tar xvf rabbitmq-server-generic-unix-3.6.15.tar
   ```

2. **启动单机RabbitMQ，生成cookie**

   ```bash
   cd /data/projects/fate/common/rabbitmq_server-3.6.15 && ./sbin/rabbitmq-server -detached
   ```

3. **修改权限**

   把cookie文件的权限设置为400：

   ```bash
   chmod -R 400 .erlang.cookie 
   ```

4. **集群部署：**

   1、多台机器都需要按照上面方式做一遍；

   2、同步cookie文件：

   ​	按照以上方式安装的话，cookie位于/home/app/.erlang.cookie

   ​	将集群中的任意一台机器上的cookie文件拷贝到其他机器上进行替换

   3、集群启动：

   ​	以mq1为基
   	（1）停mq2、mq3

   ```
    sbin/rabbitmqctl stop
   ```

   ​        （2）启动mq2、mq3

     ```
    sbin/rabbitmq-server -detached
     ```

   ​	（3）停mq2、mq3应用

      ```bash
    sbin/rabbitmqctl stop_app
      ```

   ​	（4）将mq2、mq3加到mq1中

   ​        	在mq2、mq3上执行

      ```bash
    sbin/rabbitmqctl join_cluster rabbit@mq1
      ```

   ​	（5）启动mq2、mq3应用             

   ### 四. rabbitmq配置

   1、确认集群状态

   ```bash
    rabbitmqctl cluster_status
   ```

   2、启动federation服务（enable/disable）:

   ```bash
    rabbitmq-plugins enable rabbitmq_management
    rabbitmq-plugins enable rabbitmq_federation
    rabbitmq-plugins enable rabbitmq_federation_management  
   ```

   3、添加用户

   ```bash
    rabbitmqctl add_user fate fate
   ```

   4、添加角色：

   ```bash
    rabbitmqctl set_user_tags fate administrator
   ```

   5、设置权限：

   ```bash
    rabbitmqctl set_permissions -p / fate ".*" ".*" ".*" 
   ```