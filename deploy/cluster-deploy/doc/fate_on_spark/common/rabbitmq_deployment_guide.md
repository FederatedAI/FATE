### 1. Prepare Environment

1. rabbitmq-server-generic-unix-3.6.15.tar.xz
2. otp_src_19.3.tar.gz

### 2. Install Erlang

1. Downlaod Erlang source code(otp_src_19.3.tar.gz), and extract to /data/fate/projects/common

   ```bash
   tar -zxvf otp_src_19.3.tar.gz  -C /data/projects/fate/common
   ```

2. Configure ERL_TOP

   ```bash
   cd  /data/projects/fate/common/otp_src_19.3/
   export ERL_TOP=`pwd`
   ```

3. Compile

   Use the following command:

   ```bash
   ./configure --prefix=/data/projects/fate/common/erlang
   make
   make install
   ```

   If error **No curses library functions found**raised, please install ``ncuress``; first downlaod ncurses-6.0.tar.gz

   ```bash
   tar -zxvf ncurses-6.0.tar.gz
   cd ncurses-6.0
   ./configure --with-shared --without-debug --without-ada --enable-overwrite  
   make
   make install (If error Permission denied is raised, execute with root priviliges)
   ```

4. Set Environment Variable

   After compilation, edit ERL_HOME. Edit file `/etc/profile`, add the following:

   ```bash
   cat >> /etc/profile << EOF
   export ERL_HOME=/data/projects/fate/common/erlang
   export PATH=$PATH:/data/projects/fate/common/erlang/bin
   EOF
   ```

5.  Check

   run command: erl
   
   If successfully entering Erlang environemnt, installation is success.

### 3. Install RabbitMQ

1. **Download RabbitMq Server and extract to /data/projects/common**

   ```bash
   xz -d rabbitmq-server-generic-unix-3.6.15.tar.xz
   tar xvf rabbitmq-server-generic-unix-3.6.15.tar  -C /data/projects/fate/common
   ```

2. **Start standalone RabbitMQ, generate cookie**

   ```bash
   cd /data/projects/fate/common/rabbitmq_server-3.6.15 && ./sbin/rabbitmq-server -detached
   ```

3. **Change File Permissions**

   Set cookie file permission to 400:

   ```bash
   chmod -R 400 .erlang.cookie 
   ```

4. **Cluster Deploy**

   1. Follow step 1-3 for all hosts

   2. Synchronise cookie: 

      After installation by the above steps, Erlang cookie file should locate at /home/app/.erlang.cookie

      	Copy cookie from any host in the cluster to other hosts to replace original cookies

   3. Cluster Start：

      Based on mq1

      (1) Stop Erlang node for mq2, mq3

    ```bash
    sbin/rabbitmqctl stop
    ```

   ​	(2) Start Erlang node for mq2, mq3

    ```bash
   ​ sbin/rabbitmq-server -detached
    ```

   ​	(3) Stop the RabbitMQ application on mq2、mq3

     ```bash
    sbin/rabbitmqctl stop_app
     ```

   ​	(4) Join mq2, mq3 to mq1 as a cluster

   ​   On mq2、mq3 run:

     ```bash
     sbin/rabbitmqctl join_cluster rabbit@mq1
     ```

   ​	(5) Start the RabbitMQ application mq2, mq3
   ​                     

   ### 4. rabbitmq configuration 

   1. Check cluster status

   ```bash
    rabbitmqctl cluster_status
   ```

   2. Enable federation (enable/disable):

   ```bash
    rabbitmq-plugins enable rabbitmq_management
    rabbitmq-plugins enable rabbitmq_federation
    rabbitmq-plugins enable rabbitmq_federation_management  
   ```

   3. Add user:

   ```bash
    rabbitmqctl add_user fate fate
   ```
   4. Add role:

   ```bash
    rabbitmqctl set_user_tags fate administrator
   ```

   5. Set permissions:

   ```bash
    rabbitmqctl set_permissions -p / fate ".*" ".*" ".*" 
   ```
