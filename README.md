[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![CodeStyle](https://img.shields.io/badge/Check%20Style-Google-brightgreen)](https://checkstyle.sourceforge.io/google_style.html) [![Style](https://img.shields.io/badge/Check%20Style-Black-black)](https://checkstyle.sourceforge.io/google_style.html) [![Build Status](https://travis-ci.org/FederatedAI/FATE.svg?branch=master)](https://travis-ci.org/FederatedAI/FATE)
[![codecov](https://codecov.io/gh/FederatedAI/FATE/branch/master/graph/badge.svg)](https://codecov.io/gh/FederatedAI/FATE)
[![Documentation Status](https://readthedocs.org/projects/fate/badge/?version=latest)](https://fate.readthedocs.io/en/latest/?badge=latest)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/from-referrer/)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6308/badge)](https://bestpractices.coreinfrastructure.org/projects/6308)


<div align="center">
  <img src="./doc/images/FATE_logo.png">
</div>

[DOCS](./doc) | [中文](./README_zh.md)

FATE (Federated AI Technology Enabler) is the world's first industrial grade federated learning open source framework to enable enterprises and institutions to collaborate on data while protecting data security and privacy. 
It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). 
Supporting various federated learning scenarios, FATE now provides a host of federated learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.


FATE is an open source project hosted by Linux Foundation. The [Technical Charter](https://github.com/FederatedAI/FATE-Community/blob/master/FATE_Project_Technical_Charter.pdf) sets forth the responsibilities and procedures for technical contribution to, and oversight of, the FATE (“Federated AI Technology Enabler”) Project. 

<https://fate.readthedocs.io/en/latest>

## Getting Started

### Version < 2.0
Releases history can be found in [releases](https://github.com/FederatedAI/FATE/releases), deployment resources can be found on [wiki](https://github.com/FederatedAI/FATE/wiki/Download)

### Version == 2.0.0
#### Standalone deployment
- Deploying FATE on a single node via PyPI, pre-built docker images or installers. It is for simple testing purposes. Refer to this [guide](./deploy/standalone-deploy/).

### Cluster deployment
Deploying FATE to multiple nodes to achieve scalability, reliability and manageability.

- [Cluster deployment by CLI](./deploy/cluster-deploy): Using CLI to deploy a FATE cluster.

### Quick Start
- [Training Demo With Installing FATE AND FATE-Flow From Pypi](doc/2.0/fate/quick_start.md)
- [Training Demo With Installing FATE Only From Pypi](doc/2.0/fate/ml)

## Related Repositories (Projects)
- [KubeFATE](https://github.com/FederatedAI/KubeFATE): An operational tool for the FATE platform using cloud native technologies such as containers and Kubernetes.
- [FATE-Flow](https://github.com/FederatedAI/FATE-Flow): A multi-party secure task scheduling platform for federated learning pipeline.
- [FATE-Board](https://github.com/FederatedAI/FATE-Board): A suite of visualization tools to explore and understand federated models easily and effectively.
- [FATE-Serving](https://github.com/FederatedAI/FATE-Serving): A high-performance and production-ready serving system for federated learning models.
- [FATE-Cloud](https://github.com/FederatedAI/FATE-Cloud): An infrastructure for building and managing industrial-grade federated learning cloud services.
- [EggRoll](https://github.com/WeBankFinTech/eggroll): A simple high-performance computing framework for (federated) machine learning.
- [AnsibleFATE](https://github.com/FederatedAI/AnsibleFATE): A tool to optimize and automate the configuration and deployment operations via Ansible.
- [FATE-Builder](https://github.com/FederatedAI/FATE-Builder): A tool to build package and docker image for FATE and KubeFATE.
- [FATE-Client](https://github.com/FederatedAI/FATE-Client): A tool to enable fast federated modeling tasks for FATE.
- [FATE-Test](https://github.com/FederatedAI/FATE-Test): An automated testing tool for FATE, including tests and benchmark comparisons.

## Governance 

[FATE-Community](https://github.com/FederatedAI/FATE-Community) contains all the documents about how the community members coopearte with each other. 

- [GOVERNANCE.md](https://github.com/FederatedAI/FATE-Community/blob/master/GOVERNANCE.md) documents the governance model of the project. 
- [Minutes](https://github.com/FederatedAI/FATE-Community/blob/master/meeting-minutes) of working meetings
- [Development Process Guidelines](https://github.com/FederatedAI/FATE-Community/blob/master/FederatedAI_PROJECT_PROCESS_GUIDELINE.md) 
- [Security Release Process](https://github.com/FederatedAI/FATE-Community/blob/master/SECURITY.md) 


## Getting Involved

### Contributing
FATE is an inclusive and open community. We welcome developers who are interested in making FATE better! Contributions of all kinds are welcome. Please refer to the general [contributing guideline](https://github.com/FederatedAI/FATE-Community/blob/master/CONTRIBUTING.md) of all FATE projects and the contributing guideline of each repository.

### Mailing list 

Join the FATE user [mailing list](https://groups.io/g/Fate-FedAI), and stay connected with the community and learn about the latest news and information of the FATE project. Discussion and feedback of FATE project are welcome.


### Bugs or feature requests

File bugs and features requests via the [GitHub issues](https://github.com/FederatedAI/FATE/issues). If you need help, ask your questions via the mailing list.

### Contact emails

Maintainers: FedAI-maintainers @ groups.io

Security Response Committee: FATE-security @ groups.io

### Twitter

Follow us on twitter [@FATEFedAI](https://twitter.com/FateFedAI)

### FAQ
https://github.com/FederatedAI/FATE/wiki


## License
[Apache License 2.0](LICENSE)

