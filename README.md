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
- Native [Standalone-deploy](./standalone-deploy/)
- [Deployment on Kubernetes](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy).
- Native [Cluster-deploy](./cluster-deploy).
- [Run unittest](./python/federatedml/test/)
- [Run Job with FATE-PipeLine](./doc/tutorial/pipeline/fate_client_pipeline_tutotial.rst)
- [Run Job with DSL json conf](./doc/tutorial/dsl_conf)
- [Run Job on Jupyter Notebook](./doc/tutorial/pipeline/pipeline_tutorial_0.ipynb)

## Doc
### API doc
FATE provides some API documents in [doc-api](https://fate.readthedocs.io/en/latest/?badge=latest)
### Develop Guide doc
How to develop your federated learning algorithm using FATE? you can see FATE develop guide document in [develop-guide](./doc/develop_guide.rst)
### Other doc
FATE also provides many other documents in [doc](./doc/). These documents can help you understand FATE better.

## Getting Involved

*  Join our maillist [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI). You can ask questions and participate in the development discussion.

*  For any frequently asked questions, you can check in [FAQ](https://github.com/FederatedAI/FATE/wiki).

*  Please report bugs by submitting [issues](https://github.com/FederatedAI/FATE/issues).

*  Submit contributions using [pull requests](https://github.com/FederatedAI/FATE/pulls)


### License
[Apache License 2.0](LICENSE)

