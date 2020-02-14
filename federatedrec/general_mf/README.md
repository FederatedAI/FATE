# Federated Generalized Matrix Factorization.

Popularized by the Netflix Prize, Matrix Factorization has become the defacto approach to latent factor model-based recommendation.

FATE provides an Federated GMF(Generalize Matrix Factorization) algorithm implementation using these cross-party user embeddings, while each party retains own items embeddings, then exchange their user embedding under encryption to get better performance.

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.

## Heterogeneous Generalized Matrix Factorization

The inference process of HeteroGMF is shown below:

<div style="text-align:center", align=center>
<img src="../images/FedGMF.png" alt="samples" width="500" height="300" /><br/>
Figure 1： Federated Generalized Matrix Factorization</div>

Unlike other hetero federated learning approaches, hetero MF-based methods has not need to alignment samples, instead of having similar user ids, conducting same methods to generate userIds. The sample is designed as a tuple (sample_id, user_id, item_id, rating).

In the training process, party A and party B each compute their own user and item embeddings, and send their user embeddings to arbiter party under homomorphic encryption. Arbiter then aggregates, calculates, and transfers back the final user embedding to corresponding parties. 

## Features:
1. L1 & L2 regularization
2. Mini-batch mechanism
3. Five optimization methods:
    a) “sgd”: gradient descent with arbitrary batch size
    b) “rmsprop”: RMSProp
    c) “adam”: Adam
    d) “adagrad”: AdaGrad
    e) “nesterov_momentum_sgd”: Nesterov Momentum
4. Three converge criteria:
 a) "diff": Use difference of loss between two iterations, not available for multi-host training
 b) "abs": Use the absolute value of loss
 c) "weight_diff": Use difference of model weights
5. Support multi-host modeling task. For details on how to configure for multi-host modeling task, please refer to this [guide](../../../doc/dsl_conf_setting_guide.md)
6. Support validation for every arbitrary iterations
7. Learning rate decay mechanism.