# FATE软件包构建指南

[TOC]

## 1. 软件环境

| 名称   | 版本   |
| ------ | ------ |
| JDK    | 1.8+   |
| Maven  | 3.6.1+ |
| Python | 3.6.5  |

## 2. 获取源代码

请参考[获取源代码](../common/get_source_code.zh.md)

## 3. 构建FATE系统软件包

```bash
cd FATE;
bash build/package-build/build.sh ${version_tag} all
```

**all**表示需要部署的模块，all表示所有，支持: all fate fateflow fateboard eggroll examples

## 4. 输出

```bash
ls -l FATE_install_${version}_${version_tag}.tar.gz
```

## 5. 检查FATE系统软件包

```bash
tar xzf FATE_install_${version}_${version_tag}.tar.gz;
ls -lrt FATE_install_${version}_${version_tag}
```

你可以看到下面的软件包。

| 名称             | 详情                                                      |
| ---------------- | --------------------------------------------------------- |
| bin.tar.gz       | 一些脚本                                                  |
| conf.tar.gz      | 一些配置文件                                              |
| build.tar.gz     | 构建脚本                                                  |
| deploy.tar.gz    | 部署脚本                                                  |
| example.tar.gz   | 一些算法测试实例                                          |
| fate.tar.gz      | 包括federatedml和fate arch                                |
| fateflow.tar.gz  | fateflow子系统软件包                                      |
| fateboard.tar.gz | fateboard子系统软件包                                     |
| eggroll.tar.gz   | eggroll cluster: cluster manager, node manager, rollsiter |
| fate.env         | 版本列表                                                  |
| requirements.txt | python环境所需的依赖包列表                                |
| RELEASE.md       | 发布说明                                                  |
| packages_md5.txt | 每个软件包的md5数字                                       |

## 6. 构建Python依赖安装包(可选)

你可以像这样制作python依赖安装包。

```bash
cd FATE
bash build/package-build/build.sh ${version_tag} pypi
```

产生:

```bash
FATE_install_${version}_${version_tag}/pypi.tar.gz
```

使用:

```bash
pip install -r FATE/python/requirements.txt --no-index -f FATE_install_${version}_${version_tag}/pypi
```

**确保制作依赖包的操作系统和将要安装依赖包的操作系统一致**。
**不要忘记设置${version}的值**。

## 7. 构建python环境安装包(可选)

你可以像这样制作。

```bash
cd FATE
bash build/package-build/build.sh ${version_tag} python36
```

产生:

```bash
FATE_install_${version}_${version_tag}/python36.tar.gz
```

**确保制作依赖包的操作系统和将要安装依赖包的操作系统一致**。
**不要忘记设置${version}的值**。

## 8. 构建java环境安装包(可选)

你可以像这样制作。

```bash
cd FATE
bash build/package-build/build.sh ${version_tag} jdk
```

产生:

```bash
FATE_install_${version}_${version_tag}/jdk.tar.gz
```

## 9. 构建包含FATE系统软件和环境依赖的整体包(可选)

```bash
cd FATE;
bash build/package-build/build.sh ${version_tag} bin conf examples build deploy fate fateflow fateboard eggroll proxy jdk python36 pypi
```
