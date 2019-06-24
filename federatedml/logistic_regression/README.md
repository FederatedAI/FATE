# Federated Logistic Regression

Logistic Regression(LR) is a widely used statistic model for classification problems. FATE provided two kinds of federated LR: Homogeneous LR (HomoLR) and Heterogeneous LR (HeteroLR). 

We simplified the federation process into three parties. Party A represents Guest， party B represents Host while party C, which also known as "Arbiter", is a third party that holds a private key for each party and work as a coordinator. 
 
## 1. Homogeneous LR 

As the name suggested, in HomoLR, the feature spaces of guest and hosts are identical. An optional encryption mode for computing gradients is provided for host parties. By doing this, the plain model is not available for this host any more. 

<div style="text-align:center", align=center>
<img src="./images/HomoLR.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated HomoLR Principle</div> 

The HomoLR process is shown in Figure 1. Models of Party A and Party B have the same structure.
In each iteration, each party trains its model on its own data. After that, all parties upload their encrypted (or plain, depends on your configuration) gradients to arbiter. The arbiter aggregates these gradients to form a federated gradient that will then be distributed to all parties for updating their local models. Similar to traditional LR, the training process will stop when the federated model converges or the whole training process reaches a predefined max-iteration threshold. More details is available in this [paper](https://dl.acm.org/citation.cfm?id=3133982)

## 2. Heterogeneous LR 

The HeteroLR carries out the federated learning in a different way. As shown in Figure 2, A sample alignment process is conducted before training. This sample alignment process is to identify overlapping samples stored in databases of the two involved parties. The federated model is built based on those overlapping samples. The whole sample alignment process will **not** leak confidential information (e.g., sample ids) on the two parties since it is conducted in an encrypted way. Check out [paper](https://arxiv.org/abs/1711.10677) for more details. 

 <div style="text-align:center", align=center>
<img src="./images/HeteroLR.png" alt="samples" width="500" height="300" /><br/>
Figure 2： Federated HeteroLR Principle
</div>

In the training process, party A and party B compute out the elements needed for final gradients. Arbiter aggregate them and compute
out the gradient and then transfer back to each party. Check out the [paper](https://arxiv.org/abs/1711.10677) for more details.

## 3. Heterogeneous LR with Neural Network

The heteroLR algorithm described in section 2 requires the input data to be given in tabular form. This requirement limits the application of the heteroLR algorithm. To address this issue to some extent, we extend the original heteroLR by adding neural networks in the loop. 

<div style="text-align:center", align=center>
<img src="./images/HeteroLR-NN.png" alt="architecture" width="550" height="350" />
<br/>
Figure 3: Federated HeteroLR with Neural Network Principle </div>

As shown in Figure 3, neural networks are added between the raw input data and the LR model serving as feature extractors that extract representative features from raw input data of various types. Neural networks can be CNN for processing images, RNN for processing text, autoencoder for processing general numerical vectors and many others. Currently we only support autoencoder in this algorithm. We will add other models in the near future.

## Features

Both Homo-LR and Hetero-LR supports the following features:

1. L1 and L2 regularization. (Encrypted Homo-LR do not support L1)
2. Weighted training
3. mini-batch mechanism
4. Four optimized function: 'sgd', 'rmsprop', 'adam' and 'adagrad'
5. Two converge function.
    a)	diff： Use difference of loss between two iterations to judge whether converge.
    b)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.
6. Support OneVeRest

Only Hetero-LR support:
1. Support different encrypt-mode to balance speed and security