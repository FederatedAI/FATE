
# FATE Standalone Install Package Build Guide

[TOC]

## 1. Software Environment

| Name   | Version |
| ------ | ------- |
| JDK    | 1.8+    |
| Maven  | 3.6.1+  |
| Python | 3.6.5   |

## 2. Get the source code

### 2.1 Get code from Github

```bash
git clone https://github.com/FederatedAI/FATE.git -b $branch --recurse-submodules --depth=1
```

Please set the **branch** and the **version_tag**.
The **depth** parameter represents the code that only gets the latest commit, which can speed up the clone.

### 2.2 Get code from Gitee(Try Gitee when you can't connect to Github for code)

Please reference [how_to_use_gitee](../common/how_to_use_gitee.md)

## 3. Build

```bash
cd FATE
bash build/package-build/build.sh ${version_tag}
```

The **all** means that modules needs to be deployed, all means all, supports all, python, fateboard, eggroll

## 4. Output

```bash
ls -l standalone_fate_install_${version}_${version_tag}.tar.gz
```

## 5. Check packages

```bash
tar xzf standalone_fate_install_${version}_${version_tag}.tar.gz
ls -lrt standalone_fate_install_${version}_${version_tag}
```

You can see the following package:

| Name             | Details                                                   |
| ---------------- | --------------------------------------------------------- |
| env              | environment installation packages: python36, pypi, jdk    |
| init.sh          | fate standalone initialization script                     |
| bin              | some scripts                                              |
| conf             | some configuration files                                  |
| examples         | some algorithm test examples                              |
| fate             | include federatedml and fate arch                         |
| fateflow         | include fateflow                                          |
| fateboard        | fateboard packages                                        |
| fate.env         | settings for version                                      |
| requirements.txt | necessary dependency for python environment               |
| RELEASE.md       | release document                                          |
| packages_md5.txt | md5 numbers for each package                              |

## 6. Using Standalone Install Package Install FATE Stanadlone

Please reference [Fate-standalone_deployment_guide](../deploy/../../deploy/standalone-deploy/doc/Fate-standalone_deployment_guide.md)