# FATE独立Docker软件包构建指南

[TOC]

## 1. 软件环境

| 名称   | 版本   |
| ------ | ------ |
| Docker | 18.09+ |

## 2. 获取源代码

请参考[获取源代码](../common/get_source_code.zh.md)

## 3. 构建

```bash
cd FATE;
bash build/standalon-docker-build/build.sh ${version_tag}
```

可选高级用法:

- 用自定义的源替换默认源用来安装依赖包, 安装完成后恢复系统默认

```bash
bash build/standalon-docker-build/build.sh ${version_tag} {可选, 需要替换的镜像源文件} {可选, 需要使用的pip index url}
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

请参考[standalone fate deployment guide](../deploy/../../deploy/standalone-deploy/README.zh.md)