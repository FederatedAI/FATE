[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=master)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/master/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](./examples/pipeline/README.rst) | [中文](./README_zh.md)

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support federated AI ecosystem. 
It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). 
Supporting various federated learning scenarios, FATE now provides a host of federated learning algorithms, including logistic regression, 
tree-based algorithms, deep learning and transfer learning.

<https://fate.fedai.org>


## Getting Started

### Quick Start

- [Deployment by Docker Compose](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy)
- [Native Standalone-deploy](./standalone-deploy/)
- [Deployment on Kubernetes](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy).
- [Native Cluster-deploy](./cluster-deploy).
- [Run unittest](./python/federatedml/test/)
- [Run Job with FATE-Pipeline](./doc/tutorial/pipeline/fate_client_pipeline_tutotial.rst)
- [FATE-Pipeline Tutorial in Jupyter Notebook](./doc/tutorial/pipeline/pipeline_tutorial_0.ipynb)
- [Run Job with DSL json conf](./doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.rst)

## Documentation 

### Understand FATE Design 

- [FATE structure]
- [Modules](./doc/api/federatedml/federatedml_module.rst)
- [Algorithm Parameters](./python/federatedml/param)
- [Papers & Articles](./doc/resources)

### Tools & Services

- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow)
- [FATE-Board](https://github.com/FederatedAI/FATE-Board)
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving)
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud)

### Contribute to FATE

- [develop guide](./doc/community/develop_guide.rst)
- [develop guide for FATE-Client](./doc/community/fate_client_develop_guide.rst)

### API References

- [Computing API](https://fate.readthedocs.io/en/latest/_build_temp/doc/api/computing.html)
- [Federation API](https://fate.readthedocs.io/en/latest/_build_temp/doc/api/federation.html)
- [Flow SDK API](./doc/api/flow_sdk.rst)
- [Flow Client](./doc/api/flow_client.rst)
- [Pipeline](./doc/api/pipeline.rst)

## Getting Involved

- [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI)
- [FAQ](https://github.com/FederatedAI/FATE/wiki)
- [issues](https://github.com/FederatedAI/FATE/issues)
- [pull requests](https://github.com/FederatedAI/FATE/pulls)


### License
[Apache License 2.0](LICENSE)

