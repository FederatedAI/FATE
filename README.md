[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=master)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/master/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](./examples/pipeline/README.rst) | [中文](./README_zh.md)

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

<https://fate.fedai.org>


## Federated Learning Algorithms in FATE
FATE already supports a number of federated learning algorithms, including vertical federated learning, horizontal federated learning, and federated transfer learning. More details are available in [federatedml](./python/federatedml).


## Installation

FATE can be installed on Linux or Mac. Now, FATE can support two installation approaches：

* Native installation: standalone and cluster deployments;

* Cloud native installation using KubeFATE:

	- Multi-party deployment by Docker-compose, which is for the development and testing purpose;

	- Cluster (multi-node) deployment by Kubernetes.

### Native installation:
Software environment: JDK 1.8+, Python 3.6, python virtualenv and mysql 5.6+

##### Standalone Runtime
FATE provides Standalone runtime architecture for developers. It can help developers quickly test FATE. Standalone support two types of deployment: Docker version and Manual version. Please refer to [Standalone Deployment Guide](./standalone-deploy/).

##### Cluster Runtime
FATE also provides a Cluster (distributed) runtime architecture for big data scenario. Migration from Standalone Runtime to Cluster Runtime requires only changes of the configuration. No change of the algorithm is needed. To deploy FATE on a cluster, please refer to [Cluster Deployment Guide](./cluster-deploy).

### KubeFATE installation:
Using KubeFATE, FATE can be deployed by either Docker-Compose or Kubernetes:

* For development or testing purposes, Docker-Compose is recommended. It only requires Docker environment. For more detail, please refer to [Deployment by Docker Compose](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy).

* For a production or a large scale deployment, Kubernetes is recommended as an underlying infrastructure to manage a FATE cluster. For more detail, please refer to [Deployment on Kubernetes](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy).

More instructions can be found in the repo of [KubeFATE](https://github.com/FederatedAI/KubeFATE).

## FATE-Client 
FATE-client is a tool for easy interaction with FATE. We strongly recommend you install FATE-client and use it with FATE conveniently. Please refer to this [document](./python/fate_client/README.rst) for more details on FATE-Client.


## Running Tests

A script to run all the unit tests has been provided in ./python/federatedml/test folder.

Once FATE is installed, tests can be run using:

```$ sh ./python/federatedml/test/run_test.sh```

All the unit tests should pass if FATE is installed properly.

## Documentation
### Quick Start Guide

A tutorial of getting started with modeling tasks can be found [here](./examples/pipeline/README.rst).

###  Obtaining Model and Checking out Results
Functions such as tracking component output models or logs can be invoked by a tool called fate-flow. The deployment and usage of fate-flow can be found [here](./python/fate_flow/README.md).

### API Guide
FATE provides API documents in [doc-api](https://fate.readthedocs.io/en/latest/?badge=latest).
### Development Guide
To develop your federated learning algorithms using FATE, please refer to [FATE Development Guide](./doc/develop_guide.rst).
### Other Documents
To better understand FATE, refer to documents in [doc/](./doc/). 

## Getting Involved

*  Join our maillist [FATE-FedAI Group IO](https://groups.io/g/Fate-FedAI). You can ask questions and participate in the development discussion.

*  Check out the [FAQ](https://github.com/FederatedAI/FATE/wiki) for any questions you may have.

*  Please report bugs by submitting [issues](https://github.com/FederatedAI/FATE/issues).

*  Submit contributions using [pull requests](https://github.com/FederatedAI/FATE/pulls).

* Bilibili: [@FATEFedAI](https://space.bilibili.com/457797601?from=search&seid=6776229889454067000)

* Twitter: [@FATEFedAI](https://twitter.com/FateFedAI)

### License
[Apache License 2.0](LICENSE)

