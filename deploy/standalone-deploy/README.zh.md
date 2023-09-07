# FATE 单机部署指南

[English](README.md)

## 1. 说明

**服务器配置：**

- **数量:** 1
- **配置:** 8 核 / 16GB 内存 / 500GB 硬盘
- **操作系统:** CentOS Linux release 7
- **用户:** User: app owner:apps

单机版提供 3 种部署方式，可以根据实际情况选择：

- 使用 Docker 镜像安装 FATE
- 在主机中安装 FATE (使用已编译的安装包)

## 2. 使用 Docker 镜像安装 FATE（推荐）

建议使用 Docker 镜像，这样可以大大降低遇到问题的可能性。

**注意:** 请使用实际的版本号替换示例中的 `${version}`。

### 2.1 部署前环境检查

- 主机需要能够访问外部网络，从公共网络中拉取安装包和 Docker 镜像。
- 依赖 [docker](https://download.docker.com/linux/)，Docker 建议版本为 18.09。您可以使用以下命令验证 Docker 环境：`docker --version`。有关 Docker 的起停和其他操作，请参考 `docker --help`。
- 在执行之前，请检查端口 8080 是否已被占用。如果要重新执行，请使用 Docker 命令删除以前的容器和镜像。

设置部署所需环境变量（注意，通过以下方式设置的环境变量仅在当前终端会话中有效。如果打开新的终端会话，例如重新登录或打开新窗口，请重新设置）。

```bash
export version={本次部署的 FATE 版本号, 如 2.0.0-beta}
```

示例：

```bash
export version=2.0.0-beta
```

### 2.2 拉取镜像

#### 2.2.1 通过公共镜像服务

```bash
# Docker Hub
docker pull federatedai/standalone_fate:${version}

# 腾讯云容器镜像
docker pull ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version}
docker tag ccr.ccs.tencentyun.com/federatedai/standalone_fate:${version} federatedai/standalone_fate:${version}
```

#### 2.2.2 通过镜像包

```bash
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_docker_image_${version}_release.tar.gz
docker load -i standalone_fate_docker_image_${version}_release.tar.gz
docker images | grep federatedai/standalone_fate
```

如果您能看到对应 `${version}` 的镜像，则表示镜像下载成功。

### 2.3 启动

```bash
docker run -it --name standalone_fate -p 8080:8080 federatedai/standalone_fate:${version}
```

### 2.4 测试

```bash
source /data/projects/fate/fate_flow/bin/init_env.sh
```

- [测试项](#4-测试项)

## 3. 在主机中安装 FATE（使用已编译的安装包）

**注意:** 请使用实际的版本号替换示例中的 `${version}`。

### 3.1 部署前环境检查

检查本地端口 8080、9360 和 9380 是否被占用。

```bash
netstat -apln|grep 8080;
netstat -apln|grep 9360;
netstat -apln|grep 9380
```

由于需要安装操作系统依赖包，所以需要 root 权限。您可以使用 root 用户执行后续操作，如果不使用 root 用户，请为要使用的用户分配 sudo 权限：

```bash
echo "{要使用的用户名}  ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/{要使用的用户名}
```

### 3.2 获取安装包

下载安装包并解压缩。

```bash
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_install_${version}_release.tar.gz;
tar -xzvf standalone_fate_install_${version}_release.tar.gz
```

### 3.3 安装

进入解压后的目录并使用 `bin/init.sh` 进行安装。

该脚本将自动完成以下任务：

- 安装必要的操作系统依赖包
- 安装 Python 3.6 环境
- 安装 Python 包依赖
- 安装 JDK 环境
- 配置 FATE 环境变量脚本
- 配置 FateFlow
- 配置 Fateboard
- 安装 FATE 客户端

```bash
cd standalone_fate_install_${version}_release;
bash bin/init.sh init
```

### 3.4 启动

```bash
bash bin/init.sh status
bash bin/init.sh start
```

### 3.5 测试

- 加载环境变量

```bash
source bin/init_env.sh
```

- [测试项](#4-测试项)

## 4. 测试项

### 4.1 Toy 测试

```bash
flow test toy -gid 10000 -hid 10000
```

如果成功，屏幕将显示类似下方的语句：

```bash
toy test job xxx is success
```
