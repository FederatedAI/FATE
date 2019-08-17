####  Standalone

##### Docker version

1. The host needs to be able to access the external network,pull the installation package and docker image from the public network.

2. Dependent on docker and docker-compose, You can verify the docker environment with the following command:docker --version and docker-compose --version.

3. execute follow command by root user (because need to create /var/log/fate directory).

   please follow the below step:

```
#Get code
FATE $ git clone https://github.com/WeBankFinTech/FATE.git

#Execute the command
FATE $ cd FATE/standalone-deploy
FATE $ sh build_standalone_docker.sh

#Validation results
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
FATE $ sh ./federatedml/test/run_test.sh

```

There are a few algorithms under [examples/](https://github.com/WeBankFinTech/FATE/blob/master/examples) folder, try them out!

You can also experience the fateboard access via a browser:
Http://host ip:8080.

Please ignore the following tips:

1. WARNING: Image for service mysql was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.

2. debconf: delaying package configuration, since apt-utils is not installed.

3. WARNING: You are using pip version 19.2.1, however version 19.2.2 is available.You should consider upgrading via the 'pip install --upgrade pip' command.

4. WARNING: Image for service xxx was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.

   

##### Manual install

1. Install MySQL locally and make sure that you can access it through port 3306

2. Install Python on this machine. The requirement of Python version is higher than 3.6.5 and 
   lower than 3.7. You can check the version information by python --version command, and execute
   pip --version command to see if pip can be used properly.

   ```
    python --version
    pip --version
   ```

3. Install JDK1.8 locally and check the installation success with the java -version command

   ```
   java -version
   ```

4. Check whether the local 8080 port is occupied.

   ```
   netstat -apln|grep 8080
   ```

5. Create MySQL database fate_flow and user fate_dev：

   ```
      create database fate_flow DEFAULT CHARSET utf8 COLLATE utf8_general_ci; 
      grant all on *.* to 'fate_dev'@'localhost';
      flush privileges;
   ```

6. Download the compressed package of stand-alone version and decompress it. 

  ```
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE.tar.gz
tar -xvf  FATE.tar.gz
  ```

7. Enter FATE directory and execute the init.sh.

  ```
cd FATE
sh init.sh
  ```
