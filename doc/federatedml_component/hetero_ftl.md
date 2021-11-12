# Hetero FTL

Now, hetero federated transfer learning is refactorized based on
fate-1.5. And additional test datasets are offered.

This folder contains code for implementing algorithm presented in
[\[Secure Federated Transfer Learning\].](https://arxiv.org/abs/1812.03337).

Our FTL algorithm is trying to solve problem where two participants
-host and guest - have only partial overlaps in the sample space and may
or may not have overlaps in the feature space. This is illustrated in
Figure 1. Our objective is to predict labels for host as accurately as
possible.

![Figure 1： Federated Transfer Learning in the sample and feature
space](../images/samples.png)

for two-party problem

Our solution employs an architecture of two layers: local layer and
federation layer.

![Figure 2: Architecture of Federated Transfer
Learning](../images/architecture.png)

In the Local layer, both guest and host exploit a local model for
extracting features from input data and outputting extracted features in
the form of numerical vectors. The local model can be CNN for processing
images, RNN for processing text, dense layer for processing general
numerical vectors and many others. Currently users can define their own
local layers(including nn structure, optimizer, learning rate) in the
algorithm conf.

The federation layer is for the two sides exchanging intermediate
computing components and collaboratively train the federated model by
minimizing target loss and alignment loss.

In current version, we get rid of the arbiter and still guarantee
data/model privacy. For detail on how this can be achieved, please refer
to [\[Secure Federated Transfer Learning\].](https://arxiv.org/abs/1812.03337).

We support two mode for the FTL algorithms: plain mode and encrypted
mode. In plain mode, data are computed and transferred in plaintext,
while in encrypted mode intermediate results will be encrypted using
Paillier.

What is more, in the latest Hetero FTL, we add an option: communication
efficient mode. Once communication efficient mode is enabled, for every
epoch, intermediate components are preserved and are used to conduct
several local model weights updates, thus communication cost is reduced.

## Features

  - Support plain/encrypted mode
  - Support local layer define / optimizer define
  - Support communication-efficient mode.

## Applications

Now Hetero FTL only supports binary classification.

# Quick Start

Now you can start hetero-ftl like other algorithms in FATE. Please refer
to the "examples" folder.
