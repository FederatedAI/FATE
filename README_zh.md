[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=master)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/master/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/from-referrer/)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6308/badge)](https://bestpractices.coreinfrastructure.org/projects/6308)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](doc/tutorial/pipeline/pipeline_tutorial_hetero_sbt.ipynb) | [English](./README.md)

FATE (Federated AI Technology Enabler) 是全球首个联邦学习工业级开源框架，可以让企业和机构在保护数据安全和数据隐私的前提下进行数据协作。
FATE项目使用多方安全计算 (MPC) 以及同态加密 (HE) 技术构建底层安全计算协议，以此支持不同种类的机器学习的安全计算，包括逻辑回归、基于树的算法、深度学习和迁移学习等。
FATE于2019年2月首次对外开源，并成立
[FATE TSC](https://github.com/FederatedAI/FATE-Community/blob/master/FATE_Project_Technical_Charter.pdf)
对FATE社区进行开源治理，成员包含国内主要云计算和金融服务企业。

<https://fate.readthedocs.io/en/latest>

## 教程

### 2.0以前的版本
FATE 2.0以前的版本在[发布页](https://github.com/FederatedAI/FATE/releases), 下载资源汇总页在[wiki](https://github.com/FederatedAI/FATE/wiki/Download)

### 2.0.0 版本
#### 单机版部署
在单节点上部署FATE单机版，支持从 PyPI 直接安装，docker，主机安装包三种方式。
- [单机版部署教程](./deploy/standalone-deploy)
#### 集群
- [原生集群安装](./deploy/cluster-deploy): Using CLI to deploy a FATE cluster.

### 快速开始
- [从 PyPI 下载安装 FATE 和 FATE-Flow 并启动训练任务示例](doc/2.0/fate/quick_start.md)
- [从 PyPI 下载安装 FATE，并启动训练任务示例](doc/2.0/fate/ml)

## 关联仓库
- [KubeFATE](https://github.com/FederatedAI/KubeFATE)
- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow)
- [FATE-Board](https://github.com/FederatedAI/FATE-Board)
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving)
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud)
- [FedVision](https://github.com/FederatedAI/FedVision)
- [EggRoll](https://github.com/WeBankFinTech/eggroll)
- [AnsibleFATE](https://github.com/FederatedAI/AnsibleFATE)
- [FATE-Builder](https://github.com/FederatedAI/FATE-Builder)
- [FATE-Client](https://github.com/FederatedAI/FATE-Client)
- [FATE-Test](https://github.com/FederatedAI/FATE-Test)

## 社区治理  

[FATE-Community](https://github.com/FederatedAI/FATE-Community) 仓库包含历史社区合作，沟通，会议，章程等文档。

- [GOVERNANCE.md](https://github.com/FederatedAI/FATE-Community/blob/master/GOVERNANCE.md) 
- [Minutes](https://github.com/FederatedAI/FATE-Community/blob/master/meeting-minutes) 
- [Development Process Guidelines](https://github.com/FederatedAI/FATE-Community/blob/master/FederatedAI_PROJECT_PROCESS_GUIDELINE.md) 
- [Security Release Process](https://github.com/FederatedAI/FATE-Community/blob/master/SECURITY.md) 

## 了解更多

- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)
- [Twitter: @FATEFedAI](https://twitter.com/FateFedAI)


## License
[Apache License 2.0](LICENSE)
