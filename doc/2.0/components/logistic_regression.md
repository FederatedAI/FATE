# Federated Logistic Regression

Logistic Regression(LR) is a widely used statistic model for
classification problems. FATE provided two modes of federated LR:
Homogeneous LR (HomoLR) and Heterogeneous LR (HeteroLR and Hetero_SSHE_LR).

Below lists features of each LR models:

| Linear Model  	 | Multiclass(OVR)                                                             | Multi-Host                                     	                           | Cross Validation                                                      	 | Warm-Start                                                                 |
|-----------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Hetero LR     	 | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_multi_class.py) | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_multi_host.py) | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_cv.py)      | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_warm_start.py) |
| Homo LR       	 | [&check;]()                                                                 | [&check;]()                                                                | [&check;]()                                                             | [&check;]()                                                                |

We simplified the federation process into three parties. Party A
represents Guest， party B represents Host while party C, which also
known as "Arbiter", is a third party that holds a private key for each
party and work as a coordinator.

## Heterogeneous LR

The HeteroLR carries out the federated learning in a different way. As
shown in Figure 2, A sample alignment process is conducted before
training. This sample alignment process is to identify overlapping
samples stored in databases of the two involved parties. The federated
model is built based on those overlapping samples. The whole sample
alignment process will **not** leak confidential information (e.g.,
sample ids) on the two parties since it is conducted in an encrypted
way.
ion) gradients to
arbiter. The arbiter aggregates these gradients to form a federated
gradient that will then be distributed to all parties for updating their
local models. Similar to traditional LR, the training process will stop
when the federated model converges or the whole training process reaches
a predefined max-iteration threshold. More details is available in this
[Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://dl.acm.org/citation.cfm?id=3133982).

## Coordinated LR

The CoordinatedLR carries out the federated learning in a different way. As
shown in Figure 2, A sample alignment process is conducted before
training. This sample alignment process is to identify overlapping
samples stored in databases of the two involved parties. The federated
model is built based on those overlapping samples. The whole sample
alignment process will **not** leak confidential information (e.g.,
sample ids) on the two parties since it is conducted in an encrypted
way.

![Figure 1 (Federated HeteroLR Principle)](../images/HeteroLR.png)

In the training process, party A and party B compute out the elements
needed for final gradients. Arbiter aggregate them and compute out the
gradient and then transfer back to each party. More details is available in
this: [Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption](https://arxiv.org/abs/1711.10677).

## Multi-host hetero-lr

For multi-host scenario, the gradient computation still keep the same as
single-host case. However, we use the second-norm of the difference of
model weights between two consecutive iterations as the convergence
criterion. Since the arbiter can obtain the completed model weight, the
convergence decision is happening in Arbiter.

![Figure 2 (Federated Multi-host HeteroLR
Principle)](../images/hetero_lr_multi_host.png)

## Features

- Both Homo-LR and Hetero-LR(CoordinatedLR)

> 1. L1 & L2 regularization
>
> 2. Mini-batch mechanism
>
> 3. Weighted training
>
> 4. Torch optimization methods:
     >
     >     > - rmsprop: RMSProp
     >     >   - adadelta: AdaDelta
     >     >   - adagrad: AdaGrad
     >     >   - adam: Adam
     >     >   - adamw: AdamW
     >     >   - adamax: Adamax
     >     >   - asgd: ASGD
     >     >   - nadam: NAdam
     >     >   - radam: RAdam
     >     >   - rprop: RProp
     >     >   - sgd: gradient descent with arbitrary batch size
>
> 5. Torch Learning Rate Scheduler methods:
     >     > - constant
     >     >   - step
     >     >   - linear
>
> 5. Three converge criteria:
     >
     >     > - diff  
               > >     Use difference of loss between two iterations, not available
               > >     for multi-host training;
     >     >
     >     >   - abs  
                 > >     use the absolute value of loss;
     >     >
     >     >   - weight\_diff  
                 > >     use difference of model weights
>
> 6. Support multi-host modeling task.


Hetero-LR extra features

1. When modeling a multi-host task, "weight\_diff" converge criteria is supported only.
