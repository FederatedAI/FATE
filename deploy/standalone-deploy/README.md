# FATE Single-Node Deployment Guide

[中文](./README.zh.md)

## 1. Introduction

**Server Configuration:**

- **Quantity:** 1
- **Configuration:** 8 cores / 16GB memory / 500GB hard disk
- **Operating System:** CentOS Linux release 7
- **User:** User: app owner:apps

The single-node version provides 3 deployment methods, which can be selected based on your needs:

- Install FATE from PyPI
- Install FATE using Docker Images
- Install FATE on the host machine (using pre-compiled installation packages)

## 2. Install FATE from PyPI (Recommended)

### 2.1 Installing Python Environment
- Prepare and install [conda](https://docs.conda.io/projects/miniconda/en/latest/) environment.
- Create a virtual environment:
```shell
# FATE requires Python >= 3.8
conda create -n fate_env python=3.8
conda activate fate_env
```

### 2.2 Installing FATE
This section introduces two ways to installing FATE from pypi, with and without FATE-Flow service.

#### 2.2.1 Installing FATE With FATE-FLow Service
FATE-Flow provides federated job life cycle management, includes scheduling, data management, model and metric management, etc.

##### 2.2.1.1 Installing FATE, FATE-Flow, FATE-Client
```shell
pip install fate_client[fate,fate_flow]==2.0.0.b0
```
#### 2.2.1.2 Service Initialization
```shell
fate_flow init --ip 127.0.0.1 --port 9380 --home $HOME_DIR
pipeline init --ip 127.0.0.1 --port 9380
```
- `ip`: The IP address where the service runs.
- `port`: The HTTP port the service runs on.
- `home`: The data storage directory, including data, models, logs, job configurations, and SQLite databases.

#### 2.2.1.3 Start Fate-Flow Service

```shell
fate_flow start
fate_flow status # make sure fate_flow service is started
```

FATE-Flow also provides other instructions like stop and restart, use only if users want to stop/restart fate_flow services.
```shell
# Warning: normal installing process does not need to execute stop/restart instructions.
fate_flow stop
fate_flow restart
```

#### 2.2.1.4 Testing

- [Test Items](#5-Test-Items)

### 2.2.2 Installing FATE Directly
FATE provides multiple federated algorithm and secure protocols, 
users can directly import fate and use built-in algorithms and secure protocols directly.

#### 2.2.2.1 Installing FATE
```shell
pip install pyfate==2.0.0.b0
```
#### 2.2.2.2 Using Guides
Refer to [examples](../../doc/2.0/fate/ml)


## 3. Install FATE using Docker Images

**Note:** Replace `${version}` in the examples below with the actual version number.

### 3.1 Pre-deployment Environment Check

- The host machine should have access to the external network to pull installation packages and Docker images from public networks.
- Dependency on [Docker](https://download.docker.com/linux/). Docker version 18.09 is recommended. You can verify the Docker environment using the following command: `docker --version`. For Docker start/stop and other operations, refer to `docker --help`.
- Before execution, check if port 8080 is already occupied. If you need to re-execute, use Docker commands to delete previous containers and images.

Set the necessary environment variables for deployment (note that environment variables set in this way are only valid for the current terminal session. If you open a new terminal session, such as logging in again or opening a new window, you will need to reset them).

```bash
export version={FATE version number for this deployment, e.g., 2.0.0-beta}
```

Example:

```bash
export version=2.0.0-beta
```

### 3.2 Pull Docker Images

#### 3.2.1 Via Public Image Services

```bash
# Docker Hub
docker pull federatedai/standalone_fate:${version}

# Tencent Cloud Container Image
docker pull ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version}
docker tag ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version} federatedai/standalone_fate:${version}
```

#### 3.2.2 Via Image Package

```bash
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_docker_image_${version}_release.tar.gz
docker load -i standalone_fate_docker_image_${version}_release.tar.gz
docker images | grep federatedai/standalone_fate
```

If you see an image corresponding to `${version}`, it means the image download was successful.

### 3.3 Start

```bash
docker run -it --name standalone_fate -p 8080:8080 federatedai/standalone_fate:${version}
```

### 3.4 Testing

```bash
source /data/projects/fate/fate_flow/bin/init_env.sh
```

- [Test Items](#5-Test-Items)

## 4. Install FATE on the Host Machine (Using Pre-Compiled Installation Packages)

**Note:** Replace `${version}` in the examples below with the actual version number.

### 4.1 Pre-deployment Environment Check

Check if local ports 8080, 9360, and 9380 are already occupied.

```bash
netstat -apln|grep 8080;
netstat -apln|grep 9360;
netstat -apln|grep 9380
```

Because operating system dependencies need to be installed, root privileges are required. You can execute the subsequent operations as the root user. If you don't use the root user, grant sudo privileges to the user you want to use:

```bash
echo "{username to be used}  ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/{username to be used}
```

### 4.2 Get Installation Package

Download the installation package and unpack it.

```bash
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_install_${version}_release.tar.gz;
tar -xzvf standalone_fate_install_${version}_release.tar.gz
```

### 4.3 Installation

Navigate to the unpacked directory and use `bin/init.sh` for installation.

This script will automatically:

- Install necessary operating system dependency packages
- Install Python 3.6 environment
- Install Python package dependencies
- Install JDK environment
- Configure FATE environment variable scripts
- Configure FateFlow
- Configure Fateboard
- Install FATE client

```bash
cd standalone_fate_install_${version}_release;
bash bin/init.sh init
```

### 4.4 Start

```bash
bash bin/init.sh status
bash bin/init.sh start
```

### 4.5 Testing

- Load environment variables

```bash
source bin/init_env.sh
```

- [Test Items](#4-Test-Items)

## 5. Test Items

### 5.1 Toy Test

```bash
flow test toy -gid 10000 -hid 10000
```

If successful, the screen will display statements similar to the following:

```bash
toy test job xxx is success
```
