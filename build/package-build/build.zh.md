# FATE软件包构建指南

[TOC]

## 1. 软件环境

| 名称   | 版本   |
| ------ | ------ |
| JDK    | 1.8+   |
| Maven  | 3.6.1+ |
| Python | 3.6.5  |

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
bash build/package-build/build.sh ${version_tag} all
```

**all**表示需要部署的模块，all表示所有，支持: all fate fateflow fateboard eggroll examples

## 4. 输出

```bash
ls -l FATE_install_${version}_${version_tag}.tar.gz
```

## 5. 检查软件包

```bash
tar xzf FATE_install_${version}_${version_tag}.tar.gz
ls -lrt FATE_install_${version}_${version_tag}。
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

## 6. 制作Python依赖安装包(可选)

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

## 7. 制作python环境安装包(可选)

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

## 8. 制作java环境安装包(可选)

你可以像这样制作。

```bash
cd FATE
bash build/package-build/build.sh ${version_tag} jdk
```

产生:

```bash
FATE_install_${version}_${version_tag}/jdk.tar.gz
```
