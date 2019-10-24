# Federated Linear Regression

Linear Regression(LinR) is widely used simple statistic model for predicting continuous numbers. FATE provides Heterogeneous LinR HeteroLinR). The current version of HeteroLinR also provides multi-Host support. You can specify multiple hosts in the job configuration file like the provided [examples](https://github.com/WeBankFinTech/FATE/tree/master/examples/federatedml-1.0-examples/hetero_linear_regression).

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.

## Heterogeneous LinR

The process of HeteroLinR is shown below:

<div style="text-align:center", align=center>
<img src="./images/HeteroLinR.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated HeteroLinR Principle</div>

As shown in Figure 1, a sample alignment process is conducted before training. The sample alignment process identifies overlapping samples in databases of all parties. The federated model is built based on the overlapping samples. The whole sample alignment process is conducted in encryption mode, and so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute the elements needed for final gradients. Arbiter aggregates, calculate, and transfer back the final gradients to corresponding parties. For more details on the secure model-building process, please refer to the [paper](https://arxiv.org/abs/1711.10677).

## Features:
1. L1 & L2 regularization
2. Mini-batch mechanism
3. Five optimization method:
    a)	“sgd”: gradient descent with arbitrary batch size
    b) “rmsprop”: RMSProp
    c) “adam”: Adam
    d) “adagrad”: AdaGrad
    e) “nesterov_momentum_sgd”: Nesterov Momentum
4. Three converge criteria:
    a) "diff": Use difference of loss between two iterations;
    b) "abs": use the absolute value of loss;
    c) "weight_diff": use difference of model weights
