Federated Linear Regression
===========================

Linear Regression(LinR) is a simple statistic model widely used for predicting continuous numbers. FATE provides Heterogeneous Linear Regression(HeteroLinR). HeteroLinR also supports multi-Host training. You can specify multiple hosts in the job configuration file like the provided `examples/federatedml-1.x-examples/hetero_linear_regression`.

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.


Heterogeneous LinR
------------------

The process of HeteroLinR training is shown below:


.. figure:: images/HeteroLinR.png
   :align: center
   :width: 500
   :name: linaer figure 1

   Figure 1 (Federated HeteroLinR Principle)


A sample alignment process is conducted before training. The sample alignment process identifies overlapping samples in databases of all parties. The federated model is built based on the overlapping samples. The whole sample alignment process is conducted in encryption mode, and so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute the elements needed for final gradients. Arbiter aggregates, calculates, and transfers back the final gradients to corresponding parties. For more details on the secure model-building process, please refer to this `paper. <https://arxiv.org/pdf/1902.04885.pdf>`_


Param
------

.. automodule:: federatedml.param.linear_regression_param
   :members:


Features
--------

1. L1 & L2 regularization

2. Mini-batch mechanism

3. Five optimization methods:

    :sgd: gradient descent with arbitrary batch size

    :rmsprop: RMSProp

    :adam: Adam

    :adagrad: AdaGrad

    :nesterov_momentum_sgd: Nesterov Momentum

    :stochastic quansi-newton: The algorithm details can refer to `this paper. <https://arxiv.org/abs/1912.00513v2>`_

4. Three converge criteria:

    :diff: Use difference of loss between two iterations, not available for multi-host training
    
    :abs: Use the absolute value of loss
    
    :weight_diff: Use difference of model weights

5. Support multi-host modeling task. For details on how to configure for multi-host modeling task, please refer to this `guide`_

6. Support validation for every arbitrary iterations

7. Learning rate decay mechanism

8. Support early stopping mechanism, which checks for performance change on specified metrics over training rounds. Early stopping is triggered when no improvement is found at early stopping rounds.

9. Support sparse format data as input.

10. Support stepwise. For details on stepwise mode, please refer `stepwise`_ .

.. _stepwise: ../../model_selection/stepwise/README.rst
.. _guide: ../../../doc/dsl_conf_setting_guide.rst
