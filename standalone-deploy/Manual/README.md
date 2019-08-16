# FATE standalone manual install
1. Install MySQL locally and make sure that you can access it through port 3306
2. Install Python on this machine. The requirement of Python version is higher than 3.6.5 and 
   lower than 3.7. You can check the version information by python --version command, and execute
   pip --version command to see if pip can be used properly.
   
   ````
    python --version
    pip --version
   ````
3. Install JDK1.8 locally and check the installation success with the java -version command
    ````
    java -version
    ````
4. Check whether the local 8080 port is occupied.
    ````
    netstat -apln|grep 8080
    ````
5. Create MySQL database fate_flow and user fate_dev：
    ````
   create database fate_flow DEFAULT CHARSET utf8 COLLATE utf8_general_ci; 
   grant all on *.* to 'fate_dev'@'localhost';
   flush privileges;
   ````
6. Download the compressed package of stand-alone version and decompress it
   The download link is :https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/FATE.tar.gz
   
   ````
   tar -xvf  FATE.tar.gz
   ````
7. Enter FATE directory and execute the init.sh
   ````
   cd FATE
   sh init.sh
   ````
