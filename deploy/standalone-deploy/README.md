# FATE Standalone Deployment Guide

[中文](README.zh.md)

## 1. Description

Server Configuration.

| **Number** | 1 |
| ------------ | ----------------------------------------------------- |
| **Configuration** | 8 core / 16G memory / 500G hard disk |
| **OS** | Version: CentOS Linux release 7 |
| **User** | User: app owner:apps |

The standalone version provides 3 deployment methods, which can be selected according to the actual situation.

- Install FATE using Docker image

- Install FATE on the host (using the compiled installer)

- Install FATE in the host (based on the source code compiled by the package)

## 2. Install FATE using a Docker image (recommended)

It is recommended to use a docker image, which greatly reduces the possibility of encountering problems

Note that the ${version} in the following example, please replace it with the actual version number, refer to [fate.env](../../fate.env) file for the FATE version!

### 2.1 Pre-deployment environment check

- The host needs to be able to access the external network to pull installation packages and docker images from the public network.
- Dependent on [docker](https://download.docker.com/linux/), the recommended version of docker is 18.09. You can verify the docker environment with the following command: docker --version,docker start-stop and other operations please refer to docker --help
- Before executing, please check if 8080 is already occupied. If you want to execute it again, please use the docker command to delete the previous containers and images

Set the environment variables required for deployment (note that the environment variables set in the following way are only valid for the current terminal session, if you open a new terminal session, such as a new login or a new window, please set them again)

```bash
export version={FATE version for this deployment}
```

example:

```bash
export version=1.7.0
```

### 2.2 Pulling mirrors

#### 2.2.1 Via the public mirror service

```bash
# Docker Hub
docker pull federatedai/standalone_fate:${version}

# Tencent Container Registry
docker pull ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version}
docker tag ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version} federatedai/standalone_fate:${version}
```

#### 2.2.2 Via mirror packages

   ```bash
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_docker_image_${version}_release.tar.gz
   docker load -i standalone_fate_docker_image_${version}_release.tar.gz
   docker images | grep federatedai/standalone_fate
   ```

   If you can see the image corresponding to ${version}, the image is downloaded successfully

### 2.3 Boot

   ```bash
   docker run -d --name standalone_fate -p 8080:8080 federatedai/standalone_fate:${version};
   docker ps -a | grep standalone_fate
   ```

   If you can see that the container corresponding to ${version} is running, it starts successfully

### 2.4 Testing

   - Enter the container

   ```bash
   docker exec -it $(docker ps -aqf "name=standalone_fate") bash
   source bin/init_env.sh
   ```

   - [test item](#4-test-items)

## 3. Install FATE in the host (using the compiled installer)

Note that in the following example ${version}, please replace it with the actual version number, refer to [fate.env](../../fate.env) file for the FATE version!

### 3.1 Pre-deployment environment check

Whether local ports 8080, 9360, 9380 are occupied

   ```bash
   netstat -apln|grep 8080;
   netstat -apln|grep 9360;
   netstat -apln|grep 9380
   ```

Because need to install the OS dependencies, need root privileges. You can use the root user for subsequent operations. If you do not use the root user, please use the root user to grant sudo privileges to the user you want to use:

```bash
echo "{username to use} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/{username to use}
```

### 3.2 Get the installation package

Download the installation package and unpack it

   ```bash
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_install_${version}_release.tar.gz;
   tar -xzvf standalone_fate_install_${version}_release.tar.gz
   ```

### 3.3 Installation

Go to the unpacked directory and use `bin/init.sh` to install

The script will complete automatically:

- Install the necessary OS dependencies
- Install python36 environment
- Install pypi dependencies
- Install the jdk environment
- Configure the FATE environment variable script
- Configure fateflow
- Configure fateboard
- Install the fateboard client

   ```bash
   cd standalone_fate_install_${version}_release;
   bash bin/init.sh init
   ```

### 3.4 Start

   ```bash
   bash bin/init.sh status;
   bash bin/init.sh start
   ```

### 3.5 Testing

   - Load environment variables

   ```bash
   source bin/init_env.sh
   ```

   - [test items](#4-test-items)

## 4. test items

### 4.1 Toy test

   ```bash
   flow test toy -gid 10000 -hid 10000
   ```

   If successful, the screen displays a statement similar to the following:

   ```bash
   success to calculate secure_sum, it is 2000.0
   ```

### 4.2 Unit tests

   ```bash
   fate_test unittest federatedml --yes
   ```

   If successful, the screen displays a statement like the following:

   ```bash
   there are 0 failed test
   ```

Some use case algorithms are in [examples](../../examples/dsl/v2) folder, please try using them.

You can also experience the algorithm process kanban through your browser by visiting: Http://${ip}:8080, ip is `127.0.0.1` or the actual ip of the local machine

## 5. install FATE in the host (based on the source code to compile their own package)

Please refer to [standalone fate source code deployment](./doc/standalone_fate_source_code_deployment_guide.md)
