### Building FATE Images

#### Prerequisites
1. Docker: 18
2. The host can access the Internet.

#### Building Module Images

##### Configure the Image Name
To minimize the size of images, we categorize the images into three types, they are described as follows:
- ***base Image*** with the minimum dependencies of the modules of FATE.
- ***builder Image*** contains the third-party dependencies for storage service.
- ***module Image*** contains a specified module of FATE, it is built on the top of ***base image***

Before building the images, we need to configure `.env` file to name and tag the images, so that a user can easily identify the version of codebase according to image name and tag.

A sample content of `.env` is as follows:
```bash
PREFIX=federatedai
BASE_TAG=1.0.2-release
BUILDER_TAG=1.0.2-release
TAG=1.0.2-release

# PREFIX: namespace on the registry's server.
# BASE_TAG: tag of base image.
# BUILDER_TAG: tag of builder image.
# TAG: tag of module images.
```

##### Run the Building script

Use this command to build all images:
```bash
$ bash build_cluster_docker.sh modules
```

For **Chinese developer**, it will take a long time to download the resource from the overseas server. In this case, a user could provide the `useChineseMirror` to enable the building process download resource from Chinese mirror:
```bash
$ bash build_cluster_docker.sh --useChineseMirror modules 
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
federatedai/egg                       1.0.2-release    
federatedai/fateboard                 1.0.2-release    
federatedai/serving-server            1.0.2-release     
federatedai/meta-service              1.0.2-release    
federatedai/python                    1.0.2-release     
federatedai/roll                      1.0.2-release
federatedai/proxy                     1.0.2-release
federatedai/federation                1.0.2-release
federatedai/storage-service-builder   1.0.2-release   
federatedai/base-image                1.0.2-release
```

##### Push Images to a Registry (optional)
If you need to push the images to a registry (such as Docker Hub), please use the following command to log in first: 

`$ docker login username` 

After that use this command to push the images: 

`$ bash build_cluster_docker.sh push` 

##### Package the docker images for transfer (optional)
In the machine with all docker images are ready, use the following commands to package images:
```bash
# Pull mysql and redis first if you don't have those images in your machine.
$ docker pull redis
$ docker pull mysql
$ docker save $(docker images | grep -E "redis|mysql" | awk '{print $1":"$2}') -o third-party.images.tar.gz
$ docker save $(docker images | grep federatedai| grep -v -E "base|builder" | awk '{print $1":"$2}') -o fate.images.tar.gz
```

After the `*.images.tar.gz` files were generated, they need to be transferred to the machines hosting the FATE components. In the machine with the `images.tar.gz` file, use the following command to unpackage the images:
```bash
$ docker load -i third-party.images.tar.gz
$ docker load -i fate.images.tar.gz
```

### Deployment
To deploy FATE using Docker Compose and Kubernetes, please refer to [KubeFATE](https://github.com/FederatedAI/KubeFATE) for more details
