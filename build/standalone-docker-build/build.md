# FATE Standalone Docker Package Build Guide

[TOC]

## 1. Software Environment

| Name   | Version |
| ------ | ------- |
| Docker | 18.09+  |

## 2. Get the source code

Please refer to [get source code](../common/get_source_code.md)

## 3. Build

```bash
cd FATE;
bash build/standalone-docker-build/build.sh ${version_tag}
```

Optional advanced usage:

- Replace the default source with a custom source to install dependencies, and restore the default source after installation

```bash
bash build/standalone-docker-build/build.sh ${version_tag} {Optionally, the image source file to be replaced} {Optional, the pip index url to be used}
```

## 4. Output

```bash
ls -l standalone_fate_install_${version}_${version_tag}
```

## 5. Check packages

```bash
ls -lrt standalone_fate_docker_${version}_${version_tag}
```

You can see the following package:

| Name                                                       | Details          |
| ---------------------------------------------------------- | ---------------- |
| standalone_fate_docker_image_${version}_${version_tag}.tar | docker image tar |
| fate.tar                                                   | temporary files  |

## 6. Using Standalone Docker Package Install FATE Stanadlone

Please reference [standalone fate deployment guide](../deploy/../../deploy/standalone-deploy/README.md)