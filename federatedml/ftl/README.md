### Introduction

This  folder contains code for implementing algorithm presented in [Secure Federated Transfer Learning](https://arxiv.org/abs/1812.03337). Our work is built on fate, eggroll and federation API that construct the secure, distributed and parallel infrastructure.

Our FTL algorithm is trying to solve problem where two participants - host and guest - have only partial overlaps in the sample space and may or may not have overlaps in the feature space. This is illustrated in Figure 1. Our objective is to predict labels for host as accurately as possible.

<div style="text-align:center", align=center>
<img src="./images/samples.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated Transfer Learning in the sample and feature space for two-party problem</div>


Our solution employs an architecture of two layers: local layer and federation layer.

<div style="text-align:center", align=center>
<img src="./images/architecture.png" alt="architecture" width="500" height="300" />
<br/>
Figure 2: Architecture of Federated Transfer Learning </div>


In the Local layer, both guest and host exploit a local model for extracting features from input data and outputting extracted features in the form of numerical vectors. The local model can be CNN for processing images, RNN for processing text, autoencoder for processing general numerical vectors and many others. Currently we only used autoencoder in this algorithm. We will add other models in the future.

The federation layer is for the two sides exchanging intermediate computing components and collaboratively train the federated model as presented in the algorithm. 

> Note that the nomenclature used in the code may be inconsistent with the one used in the paper. This is because the FTL algorithm is implemented as a part of the whole FATE project and thus it follows the nomenclature of this project.

We support two versions of the encrypted FTL algorithm. In the first version, we preserve data/model privacy between host and guest by delegating the work of decrypting to a third party called arbiter and assume it is trustworthy. In the second version, we get rid of the arbiter and still guarantee data/model privacy. For detail on how this can be achieved, please refer to [Secure Federated Transfer Learning](https://arxiv.org/abs/1812.03337).

### Quick Start

You can refer to *examples/hetero_ftl/README.md* to quickly start running FLT algorithm in standalone mode. You can refer to *examples/hetero_ftl/HOWTORUN.md* for more detailed information on how to run FTL algorithm.