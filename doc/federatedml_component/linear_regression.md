# Federated Linear Regression

Linear Regression(LinR) is a simple statistic model widely used for
predicting continuous numbers. FATE provides Heterogeneous Linear
Regression(HeteroLinR) and SSHE Linear Regression(HeteroSSHELinR). 

Below lists features of HeteroLinR & HeteroSSHELinR models:

| Linear Model    	| Arbiter-less Training                                                       	| Weighted Training                                                                                 	| Multi-Host                                                                                     	| Cross Validation                                                                       	| Warm-Start/CheckPoint                                                                          	|
|-----------------	|-----------------------------------------------------------------------------	|---------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------	|
| Hetero LinR     	| &cross;                                                                     	| [&check;](../../examples/pipeline/hetero_linear_regression/pipeline-hetero-linr-sample-weight.py) 	| [&check;](../../examples/pipeline/hetero_linear_regression/pipeline-hetero-linr-multi-host.py) 	| [&check;](../../examples/pipeline/hetero_linear_regression/pipeline-hetero-linr-cv.py) 	| [&check;](../../examples/pipeline/hetero_linear_regression/pipeline-hetero-linr-warm-start.py) 	|
| Hetero SSHELinR 	| [&check;](../../examples/pipeline/hetero_sshe_linr/pipeline-hetero-linr.py) 	| [&check;](../../examples/pipeline/hetero_sshe_linr/pipeline-hetero-linr-sample-weight.py)         	| &cross;                                                                                        	| [&check;](../../examples/pipeline/hetero_sshe_linr/pipeline-hetero-linr-cv.py)         	| [&check;](../../examples/pipeline/hetero_sshe_linr/pipeline-hetero-linr-warm-start.py)         	|

## Heterogeneous LinR

HeteroLinR also supports multi-Host training.
You can specify multiple hosts in the job configuration file like the
provided
<span class="title-ref">examples/dsl/v2/hetero\_linear\_regression</span>.

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

<!-- mkdocs
## Param

::: federatedml.param.linear_regression_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

## Features

1.  L1 & L2 regularization

2.  Mini-batch mechanism

3.  Five optimization methods:
    
    >   - sgd  
    >     gradient descent with arbitrary batch size
    > 
    >   - rmsprop  
    >     RMSProp
    > 
    >   - adam  
    >     Adam
    > 
    >   - adagrad  
    >     AdaGrad
    > 
    >   - nesterov\_momentum\_sgd  
    >     Nesterov Momentum
    > 
    >   - stochastic quansi-newton  
    >     The algorithm details can refer to [this paper](https://arxiv.org/abs/1912.00513v2).

4.  Three converge criteria:
    
    >   - diff  
    >     Use difference of loss between two iterations, not available
    >     for multi-host training
    > 
    >   - abs  
    >     Use the absolute value of loss
    > 
    >   - weight\_diff  
    >     Use difference of model weights

5.  Support multi-host modeling task. For details on how to configure
    for multi-host modeling task, please refer to this
    [guide](../tutorial/dsl_conf/dsl_conf_v2_setting_guide.md)

6.  Support validation for every arbitrary iterations

7.  Learning rate decay mechanism

8.  Support early stopping mechanism, which checks for performance
    change on specified metrics over training rounds. Early stopping is
    triggered when no improvement is found at early stopping rounds.

9.  Support sparse format data as input.

10. Support stepwise. For details on stepwise mode, please refer to
    [stepwise](stepwise.md) .


## Hetero-SSHE-LinR features:

1.  L1 & L2 regularization

2.  Mini-batch mechanism

3. Support different encrypt-mode to balance speed and security

4. Support early-stopping mechanism

5. Support setting arbitrary metrics for validation during training

6. Support model encryption with host model
