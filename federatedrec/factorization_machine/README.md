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

FM prediction over two parties as:
<div style="text-align:center", align=center>
<img src="./images/fig1.png" alt="samples" width="703" height="225" /><br/></div>

FM loss function over two parties is defined as:
<div style="text-align:center", align=center>
<img src="./images/fig2.png" alt="samples" width="694" height="123" /><br/></div>

where <img src="./images/fig3.png" width="65" height="27" /> is a hyper-parameter.


## Homogeneous FM

As the name suggested, in HomoFM, the feature spaces of guest and hosts are identical. An optional encryption mode for computing gradients is provided for host parties. By doing this, the plain model is not available for this host any more. 

<div style="text-align:center", align=center>
<img src="./images/HomoFM.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated HomoFM Principle</div>

The HomoFM process is shown in Figure 1. Models of Participant 1 and Participant 2 and the rest of them have the same structure.
Calculate Wp1,Vp1,Wp2,Vp2 and the rest of W and V and update the model. For more detail please refer to https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
In each iteration, each party trains its model on its own data. After that, all parties upload their plain gradients to arbiter(Homo FM do not support encrypt at this time). The arbiter aggregates these gradients to form a federated gradient that will then be distributed to all parties for updating their local models. Similar to traditional FM, the training process will stop when the federated model converges or the whole training process reaches a predefined max-iteration threshold.

## Features:

Both Homo-FM and Hetero-FM supports the following features:

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
5. Support validation for every arbitrary iterations
6. Learning rate decay mechanism.


Homo-FM support the following extra features:
1. Support multi-host modeling task. For details on how to configure for multi-host modeling task, please refer to this [guide](../../../doc/dsl_conf_setting_guide.md)
2. Secure aggregation mechanism used when more aggregating models
3. Support aggregate for every arbitrary iterations.


Hetero-FM support the following extra features:
1. Support different encrypt-mode to balance speed and security
2. Support OneVeRest