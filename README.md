# FATE
FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.


<https://www.fedai.org/>

## Getting Involved

*  Join our maillist [Fate-FedAI Group IO](https://groups.io/g/Fate-FedAI). You can ask questions and participate in the development discussion.

*  For any frequently asked questions, you can check in [FAQ](https://github.com/WeBankFinTech/FATE/wiki).  

*  Please report bugs by submitting [issues](https://github.com/WeBankFinTech/FATE/issues). 

*  Submit contributions using [pull requests](https://github.com/WeBankFinTech/FATE/pulls)


## Federated Learning Algorithms In FATE
FATE already supports a number of federated learning algorithms, including vertical federated learning, horizontal federated learning, and federated transfer learning. More details are available in [federatedml](./federatedml).


## Install
FATE can be installed on Linux or Mac. Now, FATE can support standalone and cluster deployments.FATE can be installed on Linux by using:

```
git clone https://github.com/WeBankFinTech/FATE.git
```
Software environment :jdk1.8+、Python3.6、python virtualenv、mysql5.6+、redis-5.0.2


#### Cluster
FATE also provides a distributed runtime architecture for Big Data scenario. Migration from standalone to cluster requires configuration change only. No algorithm change is needed. 

To deploy FATE on a cluster, please refer to cluster deployment guide: [cluster-deploy](./cluster-deploy). 

## Running Tests

A script to run all the unittests has been provided in ./federatedml/test folder. 

Once FATE is installed, tests can be run using:

> sh ./federatedml/test/run_test.sh

All the unittests shall pass if FATE is installed properly. 

## Example Programs


## Doc
## API doc
FATE provides some API documents in [doc-api](./doc/api/), including federatedml, eggroll, federation.
## Develop Guide doc
How to develop your federated learning algorithm using FATE? you can see FATE develop guide document in [develop-guide](./doc/develop_guide.md)
## Other doc
FATE also provides many other documents in [doc](./doc/). These documents can help you understand FATE better. 
## License
[Apache License 2.0](LICENSE)

