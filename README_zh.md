[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](doc/tutorial/pipeline/pipeline_guide.rst) | [English](./README.md)

FATE (Federated AI Technology Enabler) 是微众银行AI部门发起的开源项目，为联邦学习生态系统提供了可靠的安全计算框架。
FATE项目使用多方安全计算 (MPC) 以及同态加密 (HE) 技术构建底层安全计算协议，以此支持不同种类的机器学习的安全计算，
包括逻辑回归、基于树的算法、深度学习和迁移学习等。

FATE官方网站：<https://fate.fedai.org/>


## 教程

### 部署

#### 单机版
- [Docker Compose安装](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy)
- [原生单机版安装](../deploy/standalone-deploy/)

#### 分布式
- [Kubernetes安装](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy).
- [原生集群安装](../deploy/cluster-deploy).

### 快速开始
- [单元测试](./python/federatedml/test/)
- [使用FATE-PipeLine启动任务](./doc/tutorial/pipeline/fate_client_pipeline_tutotial.rst)
- [使用DSL json conf启动任务](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.rst)
- [在Jupyter Notebook使用FATE-Pipeline](./doc/tutorial/pipeline/pipeline_tutorial_0.ipynb)

## 文档

### 理解FATE设计

- [FATE structure]
- [组件](./doc/api/federatedml/federatedml_module.rst)
- [算法参数](./python/federatedml/param)
- [论文与资料](./doc/resources)

### 工具与服务

- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow)
- [FATE-Board](https://github.com/FederatedAI/FATE-Board)
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving)
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud)

### 贡献代码

- [开发指南](./doc/community/develop_guide.rst)
- [FATE-Client开发指南](./doc/community/fate_client_develop_guide.rst)

### API文档

- [Computing API](https://fate.readthedocs.io/en/latest/_build_temp/doc/api/computing.html)
- [Federation API](https://fate.readthedocs.io/en/latest/_build_temp/doc/api/federation.html)
- [Flow SDK API](./doc/api/flow_sdk.rst)
- [Flow Client](./doc/api/flow_client.rst)
- [Pipeline](./doc/api/pipeline.rst)
- [FATE Test](./doc/api/fate_test.rst)

## 了解更多

- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)
- [Bilibili: @FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)
- [Twitter: @FATEFedAI](https://twitter.com/FateFedAI)


### License
[Apache License 2.0](LICENSE)
