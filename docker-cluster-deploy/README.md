### Cluster Deployment by Docker Compose
FATE provides scripts to build module (i.e. federation, fateflow, egg) images. With these images, the user can easily deploy FATE on a single host by `docker-compose`.

#### Prerequisites

1. Docker: 18
2. Docker-Compose: 1.24
3. The host can access the Internet.

#### Building Module Images
If you don’t want to build images from the source, please go to “Deploying Fate” section. With the building process, each module of FATE will be packaged into a docker image so that the FATE system can be deployed on any machine flexibly.

##### Configure the Image Name
To minimize the size of images, we categorize the images into three types, they are described as follows:
- ***base Image*** with the minimum dependencies of the modules of FATE.
- ***builder Image*** contains the third-party dependencies for storage service.
- ***module Image*** contains a specified module of FATE, it is built on the top of ***base image***

Before building the images, we need to configure `.env` file to name and tag the images, so that a user can easily identify the version of codebase according to image name and tag.

A sample content of `.env` is as follows:
```bash
PREFIX=federatedai
BASE_TAG=1.0-release
BUILDER_TAG=1.0-release
TAG=1.0-release

# PREFIX: namespace on the registry's server.
# BASE_TAG: tag of base image.
# BUILDER_TAG: tag of builder image.
# TAG: tag of module images.
```

##### Run the Building script

Use this command to build and upload all images:
```bash
$ bash build_cluster_docker.sh all
```

The "Dockerfile" of all images are stored under the "./docker" sub-directory. 

While creating the `builder` image, the script will clone the third-party repo to a container and build them within the container. 

The process of building module images can be separated into 3 steps:
- Start an official "maven" container to build "jar" targets
- Run bash auto-packaging.sh script to place jar files to "example-tree-dir/"
- Build module images with files in the "example-tree-dir/"

After the command finishes, use `docker images` to check the newly generated images:
```
REPOSITORY                         TAG  
federatedai/egg                       1.0-release    
federatedai/fateboard                 1.0-release    
federatedai/serving-server            1.0-release     
federatedai/meta-service              1.0-release    
federatedai/python                    1.0-release     
federatedai/roll                      1.0-release
federatedai/proxy                     1.0-release
federatedai/federation                1.0-release
federatedai/storage-service-builder   1.0-release   
federatedai/base-image                1.0-release
```

##### Push Images to a Registry (optional)
If you need to push the images to a registry (such as Docker Hub), please use the following command to log in first: 

`$ docker log in` 

After that use this command to push the images: 

`$ bash build_cluster_docker.sh push` 


#### Deploying FATE
If you ignored the building images part, please make sure the configuration file `.env` is without any modification.

##### Configure Parties
The following steps will illustrate how to deploy two parties on different hosts.

##### Generate startup files
Before starting the FATE system, the user needs to define their parties in configuration file `./docker-configuration.sh`. 

The following sample of `docker-configuration.sh` defines two parities, they are party `10000` hosted on a machine *192.10.7.1* and `9999` hosted on a machine *192.10.7.2*.
```bash
user=root
dir=/data/projects/fate
partylist=(10000 9999)
partyiplist=(192.10.7.1 192.10.7.2)
venvdir=/data/projects/fate/venv
exchangeip=proxy

# user: The user name to log in host defined in partyiplist
```

Use the following command to deploy each party. Before running the command, ***please make sure host 192.10.7.1 and 192.10.7.2 allow password-less SSH access with SSH key, otherwise you have to input password for each host manually***:
```bash
$ bash docker-auto-deploy.sh
```

The script will copy "10000-confs.tar" and "9999-confs.tar" to host 192.10.7.1 and 192.10.7.2.

Afterward the script will log in to these hosts and use docker-compose command to start the FATE cluster.

Once the command returns, log in to any host and use `docker ps` to verify the status of cluster, an example output is as follows:

```
CONTAINER ID        IMAGE                                 COMMAND                  CREATED              STATUS              PORTS                                 NAMES
d4686d616965        federatedai/python:1.0-release           "/bin/bash -c 'sourc…"   About a minute ago   Up 52 seconds       9360/tcp, 9380/tcp                    confs-10000_python_1
fbb9d7fdbb8f        federatedai/serving-server:1.0-release   "/bin/sh -c 'java -c…"   About a minute ago   Up About a minute   6379/tcp, 8001/tcp                    confs-10000_serving-server_1
4086ef0dc2de        federatedai/fateboard:1.0-release        "/bin/sh -c 'cd /dat…"   About a minute ago   Up About a minute   0.0.0.0:8080->8080/tcp                confs-10000_fateboard_1
5cf3e1f1731a        federatedai/roll:1.0-release             "/bin/sh -c 'cd roll…"   About a minute ago   Up About a minute   8011/tcp                              confs-10000_roll_1
11c01143540b        federatedai/meta-service:1.0-release     "/bin/sh -c 'java -c…"   About a minute ago   Up About a minute   8590/tcp                              confs-10000_meta-service_1
f0976f48f0f7        federatedai/proxy:1.0-release            "/bin/sh -c 'cd /dat…"   About a minute ago   Up About a minute   0.0.0.0:9370->9370/tcp                confs-10000_proxy_1
7354af787036        redis                                 "docker-entrypoint.s…"   About a minute ago   Up About a minute   6379/tcp                              confs-10000_redis_1
ed11ce8eb20d        federatedai/egg:1.0-release              "/bin/sh -c 'cd /dat…"   About a minute ago   Up About a minute   7778/tcp, 7888/tcp, 50001-50004/tcp   confs-10000_egg_1
6802d1e2bd21        mysql                                 "docker-entrypoint.s…"   About a minute ago   Up About a minute   3306/tcp, 33060/tcp                   confs-10000_mysql_1
5386bcb7565f        federatedai/federation:1.0-release       "/bin/sh -c 'cd /dat…"   About a minute ago   Up About a minute   9394/tcp                              confs-10000_federation_1
```

##### Verify the Deployment
Since the `confs-10000_python_1` container hosts the `fate-flow` service, so we need to perform the test within that container. Use the following commands to launch:
```bash
$ docker exec -it confs-10000_python_1 bash
$ source venv/bin/activate
$ cd python/examples/toy_example/
$ python run_toy_example.py 10000 9999 1
```
If the test passed, the screen will print some messages like the follows:
```
"2019-08-29 07:21:25,353 - secure_add_guest.py[line:96] - INFO: begin to init parameters of secure add example guest"
"2019-08-29 07:21:25,354 - secure_add_guest.py[line:99] - INFO: begin to make guest data"
"2019-08-29 07:21:26,225 - secure_add_guest.py[line:102] - INFO: split data into two random parts"
"2019-08-29 07:21:29,140 - secure_add_guest.py[line:105] - INFO: share one random part data to host"
"2019-08-29 07:21:29,237 - secure_add_guest.py[line:108] - INFO: get share of one random part data from host"
"2019-08-29 07:21:33,073 - secure_add_guest.py[line:111] - INFO: begin to get sum of guest and host"
"2019-08-29 07:21:33,920 - secure_add_guest.py[line:114] - INFO: receive host sum from guest"
"2019-08-29 07:21:34,118 - secure_add_guest.py[line:121] - INFO: success to calculate secure_sum, it is 2000.0000000000002"
```
For more details about the testing result, please refer to "python/examples/toy_example/README.md" 
