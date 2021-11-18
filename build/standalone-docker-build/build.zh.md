# FATE独立Docker软件包构建指南

[TOC]

## 1. 软件环境

| 名称   | 版本   |
| ------ | ------ |
| Docker | 18.09+ |

## 2. 获取源代码

### 2.1 从Github获取代码

```bash
git clone https://github.com/FederatedAI/FATE.git -b $branch --recurse-submodules --depth=1
```

请设置**branch**参数, 若使用某个发布版本分支, 则**branch**为`v版本号`, 如`v1.7.0`
**depth**参数表示只获取最新提交的代码，这可以加快克隆的速度。

### 2.2 从Gitee获取代码（当你无法连接到Github获取代码时，可以试试Gitee）

请参考[how_to_use_gitee](../common/how_to_use_gitee.zh.md)

## 3. 构建

```bash
cd FATE
bash build/standalon-docker-build/build.sh ${version_tag}
```

## 4. 输出

```bash
ls -l standalone_fate_install_${version}_${version_tag}
```

## 5. 检查软件包

```bash
ls -lrt standalone_fate_docker_${version}_${version_tag}
```

你可以看到以下软件包。

| 名称                                                       | 详情             |
| ---------------------------------------------------------- | ---------------- |
| standalone_fate_docker_image_${version}_${version_tag}.tar | docker image tar |
| fate.tar                                                   | 临时文件         |

## 6. 使用独立的Docker包安装FATE Stanadlone

请参考[Fate-standalone_deployment_guide](../deploy/../../deploy/standalone-deploy/doc/Fate-standalone_deployment_guide.zh.md)