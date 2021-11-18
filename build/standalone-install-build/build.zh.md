# FATE单机安装包构建指南

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
**depth**参数表示只获取最新提交的代码，这可以加快克隆的速度

### 2.2 从Gitee获取代码（当你无法连接到Github获取代码时，可以试试Gitee）

请参考[how_to_use_gitee](../common/how_to_use_gitee.zh.md)

## 3. 构建

```bash
cd FATE
bash build/package-build/build.sh ${version_tag}
```

## 4. 输出

```bash
ls -l standalone_fate_install_${version}_${version_tag}.tar.gz
```

## 5. 检查软件包

```bash
tar xzf standalone_fate_install_${version}_${version_tag}.tar.gz
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

请参考[Fate-standalone_deployment_guide](../deploy/../../deploy/standalone-deploy/doc/Fate-standalone_deployment_guide.zh.md)