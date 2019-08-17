#### Standalone

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

1.  WARNING: Image for service mysql was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.
2.  debconf: delaying package configuration, since apt-utils is not installed.
3.  WARNING: You are using pip version 19.2.1, however version 19.2.2 is available.You should consider upgrading via the 'pip install --upgrade pip' command.
4.  WARNING: Image for service xxx was built because it did not already exist. To rebuild this image you must use docker-compose build or docker-compose up --build.