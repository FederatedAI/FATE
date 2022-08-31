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

### 部署
FATE 支持多种部署模式，用户可以根据自身情况进行选择。[历史发布版本可以通过这里下载](https://github.com/FederatedAI/FATE/wiki/Download)
#### 单机版
- [原生单机版安装](./deploy/standalone-deploy/)

#### 集群
- [原生集群安装](./deploy/cluster-deploy)
- [Ansible集群安装](https://github.com/FederatedAI/AnsibleFATE)
- [Kubernetes安装](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy)
- [Docker Compose安装](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy)


### 快速开始
- [使用FATE-Pipeline训练及预测纵向SBT任务](./doc/tutorial/pipeline/pipeline_tutorial_hetero_sbt.ipynb)
- [使用FATE-Pipeline构建横向NN模型](doc/tutorial/pipeline/pipeline_tutorial_homo_nn.ipynb)
- [使用DSL json conf运行任务](doc/tutorial/dsl_conf/dsl_conf_tutorial.md)
- [更多教程](doc/tutorial)

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

## 文档

### FATE设计

- [架构](./doc/architecture/README.md)
- [组件](doc/federatedml_component/README.md)
- [算法参数](./python/federatedml/param)
- [论文与会议资料](./doc/resources/README.zh.md)

### 开发资源

- [开发指南](doc/develop/develop_guide.zh.md)
- [FATE API references](doc/api)
- [Flow SDK API](doc/api/fate_client/flow_sdk.md)
- [Flow Client](https://fate-flow.readthedocs.io/en/latest/zh/fate_flow_client/)
- [FATE Pipeline](doc/api/fate_client/pipeline.md)
- [FATE Test](./doc/tutorial/fate_test_tutorial.md)
- [DSL Conf Setting Guide](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.zh.md)
- [Bilibili: @FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)

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
