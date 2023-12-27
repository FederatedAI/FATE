# Federated Logistic Regression

Logistic Regression(LR) is a widely used statistic model for
classification problems. FATE provided two modes of federated LR:
Homogeneous LR (HomoLR) and Heterogeneous LR (HeteroLR and Hetero_SSHE_LR).

Below lists features of each LR models:

| Linear Model  	 | Multiclass(OVR)                                                             | Arbiter-Less Training | Multi-Host                                     	                           | Cross Validation                                                      	 | Warm-Start                                                                 |     |
|:----------------|-----------------------------------------------------------------------------|-----------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|-----|
| Coordinated LR  | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_multi_class.py) | &cross;               | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_multi_host.py) | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_cv.py)      | [&check;](../../../examples/pipeline/coordinated_lr/test_lr_warm_start.py) |     |
| SSHE LR         | [&check;](../../../examples/pipeline/sshe_lr/test_lr_multi_class.py)        | &check;               | &cross;                                                                    | [&check;](../../../examples/pipeline/sshe_lr/test_lr_cv.py)             | [&check;](../../../examples/pipeline/sshe_lr/test_lr_warm_start.py)        |     |
| Homo LR       	 | [&check;]()                                                                 | &cross;               | [&check;]()                                                                | [&check;]()                                                             | [&check;]()                                                                |     |

We simplified the federation process into three parties. Party A
represents Guest， party B represents Host while party C, which also
known as "Arbiter", is a third party that holds a private key for each
party and work as a coordinator.

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

# Heterogeneous SSHE Logistic Regression

FATE implements a heterogeneous logistic regression without arbiter role
called for hetero_sshe_lr. More details is available in this
following paper: [When Homomorphic Encryption Marries Secret Sharing:
Secure Large-Scale Sparse Logistic Regression and Applications
in Risk Control](https://arxiv.org/pdf/2008.08753.pdf).
We have also made some optimization so that the code may not exactly
same with this paper.
The training process could be described as the
following: forward and backward process.
![Figure 3 (forward)](../images/sshe-lr_forward.png)
![Figure 4 (backward)](../images/sshe-lr_backward.png)

The training process is based secure matrix multiplication protocol(SMM),
which HE and Secret-Sharing hybrid protocol is included.
![Figure 5 (SMM)](../images/secure_matrix_multiplication.png)

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
