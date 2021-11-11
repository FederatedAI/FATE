[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](doc/tutorial/pipeline/pipeline_guide.md) | [English](./README.md)

FATE (Federated AI Technology Enabler) 是微众银行AI部门发起的开源项目，为联邦学习生态系统提供了可靠的安全计算框架。
FATE项目使用多方安全计算 (MPC) 以及同态加密 (HE) 技术构建底层安全计算协议，以此支持不同种类的机器学习的安全计算，
包括逻辑回归、基于树的算法、深度学习和迁移学习等。

FATE官方网站：<https://fate.fedai.org/>


## 教程

### 部署

#### 单机版
- [Docker Compose安装](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy)
- [原生单机版安装](./deploy/standalone-deploy/)

#### 分布式
- [Kubernetes安装](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy)
- [原生集群安装](./deploy/cluster-deploy)

### 快速开始
- [单元测试](./python/federatedml/test/)
- [使用FATE-PipeLine启动任务](./doc/tutorial/pipeline/pipeline_guide.md)
- [使用DSL json conf启动任务](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.md)
- [在Jupyter Notebook使用FATE-Pipeline](./doc/tutorial/pipeline/pipeline_tutorial_0.ipynb)

## 文档

### 理解FATE设计

- [FATE structure]
- [组件](doc/federatedml_component/index.md)
- [算法参数](./python/federatedml/param)
- [论文与资料](./doc/resources)

### 工具与服务

- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow)
- [FATE-Board](https://github.com/FederatedAI/FATE-Board)
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving)
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud)
- [FedVision](https://github.com/FederatedAI/FedVision)
- [EggRoll](https://github.com/WeBankFinTech/eggroll)
- [AnsibleFATE](https://github.com/FederatedAI/AnsibleFATE)
- [KubeFATE](https://github.com/FederatedAI/KubeFATE)

### 贡献代码

- [开发指南](doc/develop/develop_guide.zh.md)

### API文档

- [Computing API](doc/api/computing.md)
- [Federation API](./doc/api/federation.md)
- [Flow SDK API](doc/api/fate_client/flow_sdk.md)
- [Flow Client](doc/api/fate_client/flow_client.md)
- [FATE Pipeline](doc/api/fate_client/pipeline.md)
- [FATE Test](./doc/api/fate_test.md)

## 了解更多

- [FATE Community](https://github.com/FederatedAI/FATE-Community)
- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)
- [Bilibili: @FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)
- [Twitter: @FATEFedAI](https://twitter.com/FateFedAI)


### License
[Apache License 2.0](LICENSE)
