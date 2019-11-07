[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Pinpoint Satellite](https://img.shields.io/endpoint?url=https%3A%2F%2Fscan.sbrella.com%2Fadmin%2Fapi%2Fv1%2Fpinpoint%2Fshield%2FFederatedAI%2FFATE)](https://github.com/mmyjona/FATE-Serving/pulls) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) 

<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>


FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

<https://fate.fedai.org>

## Getting Involved

*  Join our maillist [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI). You can ask questions and participate in the development discussion.

*  For any frequently asked questions, you can check in [FAQ](https://github.com/FederatedAI/FATE/wiki/FATE-FAQ).  

*  Please report bugs by submitting [issues](https://github.com/FederatedAI/FATE/issues). 

*  Submit contributions using [pull requests](https://github.com/FederatedAI/FATE/pulls)


## Federated Learning Algorithms In FATE
FATE already supports a number of federated learning algorithms, including vertical federated learning, horizontal federated learning, and federated transfer learning. More details are available in [federatedml](./federatedml).


## Install
FATE can be installed on Linux or Mac. Now, FATE can support standalone and cluster deployments.

Software environment :jdk1.8+、Python3.6、python virtualenv、mysql5.6+、redis-5.0.2

#### Standalone
FATE provides Standalone runtime architecture for developers. It can help developers quickly test FATE. Standalone support two types of deployment: Docker version and Manual version. Please refer to Standalone deployment guide: [standalone-deploy](./standalone-deploy/)

#### Cluster
FATE also provides a distributed runtime architecture for Big Data scenario. Migration from standalone to cluster requires configuration change only. No algorithm change is needed. 

To deploy FATE on a cluster, please refer to cluster deployment guide: [cluster-deploy](./cluster-deploy).

#### Get source
```shell
git clone --recursive git@github.com:FederatedAI/FATE.git
```

## Running Tests

A script to run all the unittests has been provided in ./federatedml/test folder. 

Once FATE is installed, tests can be run using:

> sh ./federatedml/test/run_test.sh

All the unittests shall pass if FATE is installed properly. 

## Example Programs

### Quick Start

We have provided a python script for quick starting modeling task. This scrip is located at ["examples/federatedml-1.x-examples"](./examples/federatedml-1.x-examples)

#### Standalone Version
1. Start standalone version hetero-lr task (default)
> python quick_run.py


#### Cluster Version

1. Host party:
> python quick_run.py -r host

This is just uploading data

2. Guest party:
> python quick_run.py -r guest

The config files that generated is stored in a new created folder named **user_config**

#### Start a Predict Task
Once you finish one training task, you can start a predict task. You need to modify "TASK" variable in quick_run.py script as "predict":
```
# Define what type of task it is
# TASK = 'train'
TASK = 'predict'
```
Then all you need to do is running the following command:
> python quick_run.py

Please note this works only if you have finished the trainning task.

###  Obtain Model and Check Out Results
We provided functions such as tracking component output models or logs etc. through a tool called fate-flow. The deployment and usage of fate-flow can be found [here](./fate_flow/README.md)


## Doc
### API doc
FATE provides some API documents in [doc-api](./doc/api/), including federatedml, eggroll, federation.
### Develop Guide doc
How to develop your federated learning algorithm using FATE? you can see FATE develop guide document in [develop-guide](./doc/develop_guide.md)
### Other doc
FATE also provides many other documents in [doc](./doc/). These documents can help you understand FATE better.
### License
[Apache License 2.0](LICENSE)

