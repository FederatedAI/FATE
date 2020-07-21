[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=develop-1.4)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/develop-1.4/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOC](./doc) | [Quick Start](./examples/federatedml-1.x-examples) | [中文](./README_zh.md)

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

<https://fate.fedai.org>


## Federated Learning Algorithms In FATE
FATE already supports a number of federated learning algorithms, including vertical federated learning, horizontal federated learning, and federated transfer learning. More details are available in [federatedml](./federatedml).


## Install

FATE can be installed on Linux or Mac. Now, FATE can support：

* Native installation: standalone and cluster deployments;

* KubeFATE installation:

	- Multipal parties deployment by docker-compose, which for devolopment and test purpose;

	- Cluster (multi-node) deployment by Kubernetes

### Native installation:
Software environment :jdk1.8+、Python3.6、python virtualenv、mysql5.6+、redis-5.0.2

##### Standalone
FATE provides Standalone runtime architecture for developers. It can help developers quickly test FATE. Standalone support two types of deployment: Docker version and Manual version. Please refer to Standalone deployment guide: [standalone-deploy](./standalone-deploy/)

##### Cluster
FATE also provides a distributed runtime architecture for Big Data scenario. Migration from standalone to cluster requires configuration change only. No algorithm change is needed.

To deploy FATE on a cluster, please refer to cluster deployment guide: [cluster-deploy](./cluster-deploy).


### KubeFATE installation:
Using KubeFATE, FATE can be deployed by either docker-compose or Kubernetes:

* For development or testing purposes, docker-compose is recommended. It only requires Docker enviroment. For more detail, please refer to [Deployment by Docker Compose](https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy).

* For a production or a large scale deployment, Kubernetes is recommended as an underlying infrastructure to manage FATE system. For more detail, please refer to [Deployment on Kubernetes](https://github.com/FederatedAI/KubeFATE/blob/master/k8s-deploy).

More instructions can be found in [KubeFATE](https://github.com/FederatedAI/KubeFATE).

## Running Tests

A script to run all the unittests has been provided in ./federatedml/test folder.

Once FATE is installed, tests can be run using:

> sh ./federatedml/test/run_test.sh

All the unittests shall pass if FATE is installed properly.

## Example Programs

### Quick Start

We have provided a python script for quick starting modeling task. This scrip is located at ["examples/federatedml-1.x-examples"](./examples/federatedml-1.x-examples)

###  Obtain Model and Check Out Results
We provided functions such as tracking component output models or logs etc. through a tool called fate-flow. The deployment and usage of fate-flow can be found [here](./fate_flow/README.rst)


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

