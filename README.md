[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=master)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/master/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](doc/tutorial/pipeline/pipeline_tutorial_hetero_sbt.ipynb) | [中文](./README_zh.md)

FATE (Federated AI Technology Enabler) is the world's first industrial grade federated learning open source framework initiated by Webank's AI Department to enable enterprises and institutions to collaborate on data while protecting data security and privacy. 
It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). 
Supporting various federated learning scenarios, FATE now provides a host of federated learning algorithms, including logistic regression, tree-based algorithms, deep learning, and transfer learning.
FATE open-sourced for the first time in February 2019, and was donated to Linux Foundation by WeBank in June 2019. 
[FATE TSC](https://github.com/FederatedAI/FATE-Community/blob/master/FATE_Project_Technical_Charter.pdf) was established 
to lead FATE open-source community, with members from major domestic cloud computing and financial service enterprises. 

<https://fate.readthedocs.io/en/latest>


## Getting Started

### Deploy

#### Standalone
- [Native Standalone-deploy](./deploy/standalone-deploy/)

#### Cluster
- [Native Cluster-deploy](./deploy/cluster-deploy)
- [Deployment by Ansible](https://github.com/FederatedAI/AnsibleFATE)
- [Deployment by Docker Compose](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy)
- [Deployment on Kubernetes](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy)

### Quick Start
- [Train & Predict Hetero SecureBoost with FATE-Pipeline](./doc/tutorial/pipeline/pipeline_tutorial_hetero_sbt.ipynb)
- [Build Homo NN model with FATE-Pipeline](doc/tutorial/pipeline/pipeline_tutorial_homo_nn.ipynb)
- [Run Job with DSL json conf](doc/tutorial/dsl_conf/dsl_conf_tutorial.md)
- [Tutorials](doc/tutorial)

## Documentation 

### FATE Design 

- [Architecture](./doc/architecture/README.md)
- [Components](./doc/federatedml_component/README.md)
- [Algorithm Parameters](./python/federatedml/param)
- [Paper & Conference](./doc/resources/README.md)

### Association Repository

- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow)
- [FATE-Board](https://github.com/FederatedAI/FATE-Board)
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving)
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud)
- [FedVision](https://github.com/FederatedAI/FedVision)
- [EggRoll](https://github.com/WeBankFinTech/eggroll)
- [AnsibleFATE](https://github.com/FederatedAI/AnsibleFATE)
- [KubeFATE](https://github.com/FederatedAI/KubeFATE)

### Contribute to FATE

- [develop guide](doc/develop/develop_guide.md)

### API References
- [Session API](doc/api/session.md)
- [Computing API](doc/api/computing.md)
- [Federation API](./doc/api/federation.md)
- [Flow SDK API](doc/api/fate_client/flow_sdk.md)
- [Flow Client](https://fate-flow.readthedocs.io/en/latest/en/fate_flow_client/)
- [FATE Pipeline](doc/api/fate_client/pipeline.md)
- [FATE Test](./doc/tutorial/fate_test_tutorial.md)
- [DSL Conf Setting Guide](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.md)

### Online Courses
- [Bilibili: @FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)

## Getting Involved

- [FATE Community](https://github.com/FederatedAI/FATE-Community)
- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)
- [Twitter: @FATEFedAI](https://twitter.com/FateFedAI)


## License
[Apache License 2.0](LICENSE)

