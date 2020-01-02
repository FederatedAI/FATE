# Federated Factorization Machine

Factorization Machine(FM) is a  supervised learning approach incorporates second-order feature interaction.
Federated factorization machine computes these cross-party cross-features and their gradients under encryption. 

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.

## Heterogeneous FM

The inference process of HeteroFM is shown below:

<div style="text-align:center", align=center>
<img src="./images/HeteroFM.png" alt="samples" width="500" height="300" /><br/>
Figure 1： Federated HeteroFM</div>

Similar to other hetero federated learning approch, a sample alignment process is conducted before training. The sample alignment process identifies overlapping samples in databases of all parties. The federated model is built based on the overlapping samples. The whole sample alignment process is conducted in encryption mode, and so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute their own linear and cross-features forward results, and compute sucure cross-party cross-features under homomorphic encryption. Arbiter then aggregates, calculates, and transfers back the final gradients to corresponding parties. 

## Features:
1. L1 & L2 regularization
2. Mini-batch mechanism
3. Five optimization methods:
    a)	“sgd”: gradient descent with arbitrary batch size
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