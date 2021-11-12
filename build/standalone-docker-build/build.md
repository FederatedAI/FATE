
# FATE Standalone Docker Package Build Guide

[TOC]

## 1. Software Environment

| Name   | Version |
| ------ | ------- |
| JDK    | 1.8+    |
| Maven  | 3.6.1+  |
| Python | 3.6.5   |

## 2. Build

```bash
git clone https://github.com/FederatedAI/FATE.git -b ${branch} --recurse-submodules --depth=1
cd FATE
bash build/standalone-docker-build/build.sh ${version_tag}
```

Please set the **branch** and the **version_tag**.
The **depth** parameter represents the code that only gets the latest commit, which can speed up the clone.
The **all** means that modules needs to be deployed, all means all, supports all, python, fateboard, eggroll

## 3. Output

```bash
ls -l standalone_fate_install_${version}
```

## 4. Check packages

```bash
ls -lrt standalone_fate_docker_${version}
```

You can see the following package:

| Name             | Details                                                   |
| ---------------- | --------------------------------------------------------- |
| standalone_fate_docker_image_${version}_release.tar              | docker image tar |
| fate.tar                        | temporary files|

## 5. Using Standalone Docker Package Install FATE Stanadlone

Please reference [Fate-standalone_deployment_guide](../deploy/../../deploy/standalone-deploy/doc/Fate-standalone_deployment_guide.md)