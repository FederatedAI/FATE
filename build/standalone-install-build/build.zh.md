# FATE单机安装包构建指南

[TOC]

## 1. 软件环境

| 名称   | 版本   |
| ------ | ------ |
| JDK    | 1.8+   |
| Maven  | 3.6.1+ |
| Python | 3.6.5  |

## 2. 获取源代码

请参考[获取源代码](../common/get_source_code.zh.md)

## 3. 构建

```bash
cd FATE;
bash build/package-build/build.sh ${version_tag}
```

## 4. 输出

```bash
ls -l standalone_fate_install_${version}_${version_tag}.tar.gz
```

## 5. 检查软件包

```bash
tar xzf standalone_fate_install_${version}_${version_tag}.tar.gz;
ls -lrt standalone_fate_install_${version}_${version_tag}
```

你可以看到以下软件包。

| 名称             | 详情                             |
| ---------------- | -------------------------------- |
| env              | 环境安装包： python36, pypi, jdk |
| init.sh          | 初始化脚本                       |
| bin              | 一些脚本                         |
| conf             | 一些配置文件                     |
| 例子             | 一些算法的测试例子               |
| 命运             | 包括federatedml和fate arch       |
| fateflow         | fateflow软件包                   |
| fateboard        | fateboard软件包                  |
| fate.env         | 版本列表                         |
| requirements.txt | python依赖列表                   |
| RELEASE.md       | 发布说明                         |
| packages_md5.txt | 每个包的md5数字                  |

## 6. 使用独立安装包安装FATE Stanadlone

请参考[standalone fate deployment guide](../deploy/../../deploy/standalone-deploy/README.zh.md)