#### Standalone

##### Docker version

1. The host needs to be able to access the external network,pull the installation package and docker image from the public network.
2. Dependent on docker and docker-compose, you need to install and configure first.
3. Need to create /var/log/fate directory, so need to be executed by the root user.

```
FATE $ git clone https://github.com/WeBankFinTech/FATE.git
FATE $ cd FATE/standalone-deploy
FATE $ sh build_standalone_docker.sh
FATE $ CONTAINER_ID=`docker ps -aqf "name=fate_python"`
FATE $ docker exec -t -i ${CONTAINER_ID} bash
```

There are a few algorithms under [examples/](https://github.com/WeBankFinTech/FATE/blob/master/examples) folder, try them out!

You can also experience the fateboard access via a browser:
Http://host ip:8080