[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](doc/tutorial/pipeline/pipeline_tutorial_hetero_sbt.ipynb) | [English](./README.md)

FATE (Federated AI Technology Enabler) 是微众银行AI部门发起的全球首个联邦学习工业级开源框架，可以让企业和机构在保护数据安全和数据隐私的前提下进行数据协作。
FATE项目使用多方安全计算 (MPC) 以及同态加密 (HE) 技术构建底层安全计算协议，以此支持不同种类的机器学习的安全计算，包括逻辑回归、基于树的算法、深度学习和迁移学习等。
FATE于2019年2月首次对外开源，并于2019年6月由微众银行捐献给Linux基金会，并成立
[FATE TSC](https://github.com/FederatedAI/FATE-Community/blob/master/FATE_Project_Technical_Charter.pdf)
对FATE社区进行开源治理，成员包含国内主要云计算和金融服务企业。

<https://fate.readthedocs.io/en/latest>

## 教程

### 部署

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

## 文档

### FATE设计

- [架构](./doc/architecture/README.md)
- [组件](doc/federatedml_component/README.md)
- [算法参数](./python/federatedml/param)
- [论文与会议资料](./doc/resources/README.zh.md)

### 关联仓库

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
- [Session API](doc/api/session.md)
- [Computing API](doc/api/computing.md)
- [Federation API](./doc/api/federation.md)
- [Flow SDK API](doc/api/fate_client/flow_sdk.md)
- [Flow Client](https://github.com/FederatedAI/FATE-Flow/blob/develop-1.7.0/doc/fate_flow_client.md)
- [FATE Pipeline](doc/api/fate_client/pipeline.md)
- [FATE Test](./doc/tutorial/fate_test_tutorial.md)
- [DSL Conf Setting Guide](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.zh.md)

### 在线课程
- [Bilibili: @FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)


## 了解更多

- [FATE Community](https://github.com/FederatedAI/FATE-Community)
- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)
- [Twitter: @FATEFedAI](https://twitter.com/FateFedAI)


## License
[Apache License 2.0](LICENSE)
