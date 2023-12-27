# Federated Linear Regression

Linear Regression(LinR) is a simple statistic model widely used for
predicting continuous numbers. FATE provides Heterogeneous Linear
Regression(CoordinatedLinR).

Below lists features of Coordinated LinR model:

| Linear Model    	 | Multi-Host                                                                   	 | Cross Validation                                                 	     | Warm-Start                                                                     |
|-------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Hetero LinR     	 | [&check;](../../../examples/pipeline/coordinated_linr/test_linr_multi_host.py) | [&check;](../../../examples/pipeline/coordinated_linr/test_linr_cv.py) | [&check;](../../../examples/pipeline/coordinated_linr/test_linr_warm_start.py) |
| SSHE LinR         | &cross;                                                                        | [&check;](../../../examples/pipeline/sshe_linr/test_linr_cv.py)        | [&check;](../../../examples/pipeline/sshe_linr/test_linr_warm_start.py)        |                                                  |                                                                        |                                                               |

## Coordinated LinR

CoordinatedLinR also supports multi-Host training.

Here we simplify participants of the federation process into three
parties. Party A represents Guest, party B represents Host. Party C,
which is also known as “Arbiter,” is a third party that works as
coordinator. Party C is responsible for generating private and public
keys.

The process of HeteroLinR training is shown below:

![Figure 1 (Federated HeteroLinR
Principle)](../images/HeteroLinR.png)

A sample alignment process is conducted before training. The sample
alignment process identifies overlapping samples in databases of all
parties. The federated model is built based on the overlapping samples.
The whole sample alignment process is conducted in encryption mode, and
so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute the elements
needed for final gradients. Arbiter aggregates, calculates, and
transfers back the final gradients to corresponding parties. For more
details on the secure model-building process, please refer to this
[paper.](https://arxiv.org/pdf/1902.04885.pdf)

## Features

1. L1 & L2 regularization

2. Mini-batch mechanism

3. Weighted training

4. Torch optimization methods:

   > - rmsprop: RMSProp
   >   - adadelta: AdaDelta
   >   - adagrad: AdaGrad
   >   - adam: Adam
   >   - adamw: AdamW
   >   - adamax: Adamax
   >   - asgd: ASGD
   >   - nadam: NAdam
   >   - radam: RAdam

> - rprop: RProp
    >     - sgd: gradient descent with arbitrary batch sizegorithm details can refer
    to [this paper](https://arxiv.org/abs/1912.00513v2).

5. Torch Learning Rate Scheduler methods:
   > - constant
   >   - step
   >   - linear
6. Three converge criteria:

   > - diff  
       >     Use difference of loss between two iterations, not available
       >     for multi-host training
   >
   >   - abs  
         >     Use the absolute value of loss
   >
   >   - weight\_diff  
         >     Use difference of model weights

5. Support multi-host modeling task.

## Hetero-SSHE-LinR features:

1. Mini-batch mechanism

2. Support early-stopping mechanism

3. Support setting arbitrary frequency for revealing loss 
