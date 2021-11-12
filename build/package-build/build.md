
# FATE Packages Build Guide

[TOC]

## 1. Software Environment

| Name   | Version |
| ------ | ------- |
| JDK    | 1.8+    |
| Maven  | 3.6.1+  |
| Python | 3.6.5   |

## 2. Build

```bash
git clone https://github.com/FederatedAI/FATE.git -b $branch --recurse-submodules --depth=1
cd FATE
bash build/package-build/build.sh $version_tag all
```

Please set the **branch** and the **version_tag**.
The **depth** parameter represents the code that only gets the latest commit, which can speed up the clone.
The **all** means that modules needs to be deployed, all means all, supports all, python, fateboard, eggroll

## 3. Output

```bash
ls -l FATE_install_$version-$version_tag.tar.gz
```

## 4. Check packages

```bash
tar xzf FATE_install_$version-$version_tag.tar.gz
ls -lrt FATE_install_$version
```

You can see the following package:

| Name             | Details                                                   |
| ---------------- | --------------------------------------------------------- |
| bin.tar.gz       | some scripts                                              |
| conf.tar.gz      | some configuration files                                  |
| build.tar.gz     | build scripts                                             |
| deploy.tar.gz    | deploy scripts                                            |
| examples.tar.gz  | some algorithm test examples                              |
| fate.env         | settings for version                                      |
| requirements.txt | necessary dependency for python environment               |
| fate.tar.gz      | include federatedml and fate arch                         |
| fateflow.tar.gz  | include fateflow                                          |
| fateboard.tar.gz | fateboard packages                                        |
| eggroll.tar.gz   | eggroll cluster: cluster manager, node manager, rollsiter |
| RELEASE.md       | release document                                          |
| packages_md5.txt | md5 numbers for each package                              |

## 6. Make python requirements package

You can make python requirements package like:

```bash
pip download -r FATE/python/requirements.txt -d pip-packages-fate-$version/
```

And then you can use it like:

```bash
pip install -r FATE/python/requirements.txt --no-index --find-links=pip-packages-fate-$version/
```

**Don't forget to set the value of $version**
