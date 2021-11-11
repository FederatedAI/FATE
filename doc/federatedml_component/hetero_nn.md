# Heterogeneous Neural Networks

Neural networks are probably the most popular machine learning
algorithms in recent years. FATE provides a federated Heterogeneous
neural network implementation.

This federated heterogeneous neural network framework allows multiple
parties to jointly conduct a learning process with partially overlapping
user samples but different feature sets, which corresponds to a
vertically partitioned virtual data set. An advantage of Hetero NN is
that it provides the same level of accuracy as the non
privacy-preserving approach while at the same time, reveal no
information of each private data provider.

## Basic FrameWork

The following figure shows the proposed Federated Heterogeneous Neural
Network framework.

![Figure 1 (Framework of Federated Heterogeneous Neural
Network)](../images/hetero_nn_framework.png)

Party B: We define the party B as the data provider who holds both a
data matrix and the class label. Since the class label information is
indispensable for supervised learning, there must be an party with
access to the label y. The party B naturally takes the responsibility as
a dominating server in federated learning.

Party A: We define the data provider which has only a data matrix as
party A. Party A plays the role of clients in the federated learning
setting.

The data samples are aligned under an encryption scheme. By using the
privacy-preserving protocol for inter-database intersections, the
parties can find their common users or data samples without compromising
the non-overlapping parts of the data sets.

Party B and party A each have their own bottom neural network model,
which may be different. The parties jointly build the interactive layer,
which is a fully connected layer. This layer's input is the
concatenation of the two parties' bottom model output. In addition, only
party B owns the model of interactive layer. Lastly, party B builds the
top neural network model and feeds the output of interactive layer to
it.

## Forward Propagation of Federated Heterogeneous Neural Network

Forward Propagation Process consists of three parts.

  - Part Ⅰ  
    Forward Propagation of Bottom Model.

> 1.  Party A feeds its input features X to its bottom model and gets
>     the forward output of bottom model alpha\_A
> 2.  Party B feeds its input features X to its bottom model and gets
>     the forward output of bottom model alpha\_B if active party has
>     input features.

  - Part ⅠⅠ  
    Forward Propagation of Interactive Layer.

> 1.  Party A uses additive homomorphic encryption to encrypt
>     alpha\_A(mark as \[alpha \_A\] ), and sends the encrypted result
>     to party B.
> 2.  Party B receives the \[alpha\_A\], multiplies it by interactive
>     layer's party A model weight W\_A, get \[z\_A\]. Party B also
>     multiplies its interactive layer's weight W\_B by its own bottom
>     output, getting z\_B. Party B generates noise epsilon\_B, adds it
>     to \[z\_A\] and sends addition result to party A.
> 3.  Party A calculates the product of accumulate noise epsilon\_acc
>     and bottom input alpha\_A (epsilon\_acc \* alpha\_A). Decrypting
>     the received result \[z\_A + epsilon\_B\], Party A adds the
>     product to it and sends result to Active party.
> 4.  Party B subtracts the party A's sending value by epsilon\_B( get
>     z\_A + epsilon\_acc \* alpha\_A), and feeds z = z\_A +
>     epsilon\_acc \* alpha\_A + z\_B(if exists) to activation function.

  - Part ⅠⅠⅠ  
    Forward Propagation of Top Model.

> 1.  Party B takes the output of activation function's output of
>     interactive layer g(z) and runs the forward process of top model.
>     The following figure shows the forward propagation of Federated
>     Heterogeneous Neural Network framework.
> 
> ![Figure 2 (Forward Propagation of Federated Heterogeneous Neural
> Network)](../images/hetero_nn_forward_propagation.png)

## Backward Propagation of Federated Heterogeneous Neural Network

Backward Propagation Process also consists of three parts.

  - Part I  
    Backward Propagation of Top Model.

> 1.  Party B calculates the error delta of interactive layer output,
>     then updates top model.

  - Part II  
    Backward Propagation of Interactive layer.

> 1.  Party B calculates the error delta\_act of activation function's
>     output by delta.
> 2.  Party B propagates delta\_bottomB = delta\_act \* W\_B to bottom
>     model, then updates W\_B(W\_B -= eta \* delta\_act \* alpha\_B).
> 3.  Party B generates noise epsilon\_B, calculates \[delta\_act \*
>     (alpha\_A + epsilon\_B\] and sends it to party A.
> 4.  Party A encrypts epsilon\_acc, sends \[epsilon\_acc\] to party B.
>     Then party B decrypts the received value. Party A generates noise
>     epsilon\_A, adds epsilon\_A / eta to decrypted result(delta\_act
>     \* alpha\_A + epsilon\_B + epsilon\_A / eta) and add epsilon\_A to
>     accumulate noise epsilon\_acc(epsilon\_acc += epsilon\_A). Party A
>     sends the addition result to party B. (delta\_act \* W\_A +
>     epsilon\_B + epsilon\_A / eta)
> 5.  Party B receives \[epsilon\_acc\] and delta\_act \* alpha\_A +
>     epsilon\_B + epsilon\_A / eta. Firstly it sends party A's bottom
>     model output' error \[delta\_act \* W\_A + acc\] to party A.
>     Secondly updates W\_A -= eta \* (delta\_act \* W\_A + epsilon\_B +
>     epsilon\_A / eta - epsilon\_B) = eta \* delta\_act \* W\_A -
>     epsilon\_B = W\_TRUE - epsilon\_acc. Where W\_TRUE represents the
>     actually weights.
> 6.  Party A decrypts \[delta\_act \* (W\_A + acc)\] and passes
>     delta\_act \* (W\_A + acc) to its bottom model.

  - Part III  
    Backward Propagation of Bottom Model.

> 1.  Party B and party A updates their bottom model separately. The
>     following figure shows the backward propagation of Federated
>     Heterogeneous Neural Network framework.
> 
> ![Figure 3 (Backward Propagation of Federated Heterogeneous Neural
> Network)](../images/hetero_nn_backward_propagation.png)

<!-- mkdocs
## Param

::: federatedml.param.hetero_nn_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

## Other features

  - Allow party B's training without features.
  - Support evaluate training and validate data during training process
  - Support use early stopping strategy since FATE-v1.4.0
  - Support selective backpropagation since FATE-v1.6.0
  - Support low floating-point optimization since FATE-v1.6.0
  - Support drop out strategy of interactive layer since FATE-v1.6.0

[1] Zhang Q, Wang C, Wu H, et al. GELU-Net: A Globally Encrypted,
Locally Unencrypted Deep Neural Network for Privacy-Preserved
Learning\[C\]//IJCAI. 2018: 3933-3939.

[2] Zhang Y, Zhu H. Additively Homomorphical Encryption based Deep
Neural Network for Asymmetrically Collaborative Machine Learning\[J\].
arXiv preprint arXiv:2007.06849, 2020.
