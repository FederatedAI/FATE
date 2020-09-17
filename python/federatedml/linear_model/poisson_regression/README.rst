Federated Poisson Regression
============================

Poisson distribution is a convenient model for modeling occurrences within a certain time period or geographical area. It is commonly used for predicting rates of low-frequency events. FATE provides Heterogeneous Poisson Regression(HeteroPoisson). The module can accept data with exposure variable, whose column name may be specified in job configuration file. Please refer to the `examples/federatedml-1.x-examples/hetero_poisson_regression` on how to specify exposure element in job configuration.

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.

Heterogeneous Poisson
---------------------

The process of HeteroPoisson training is shown below:

.. figure:: images/HeteroPoisson.png
   :width: 500
   :name: possion figure 1
   :align: center

   Figure 1： Federated HeteroPoisson Principle

A sample alignment process is conducted before training. The sample alignment process identifies overlapping samples in databases of all parties. The federated model is built based on the overlapping samples. The whole sample alignment process is conducted in encryption mode, and so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute the elements needed for final gradients. Arbiter aggregates, calculates, and transfers back the final gradients to corresponding parties. Arbiter also decides at the end of each iteration whether the model has converged, based on the stopping criteria set by Guest.


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

4. Three converge criteria:

    :diff: Use difference of loss between two iterations

    :abs: Use the absolute value of loss

    :weight_diff: Use difference of model weights

5. Support use of exposure variable. Guest party may specify "exposure_colname" in the job configuration file

6. Support validation for every arbitrary iterations

7. Learning rate decay mechanism

8. Support early stopping mechanism, which checks for performance change on specified metrics over training rounds. Early stopping is triggered when no improvement is found at early stopping rounds.

9. Support sparse format data as input.

10. Support stepwise. For details on stepwise mode, please refer `[here]. <../../model_selection/stepwise/README.rst>`_


.. note::

	The performance of poisson regression is highly dependent on model meta and the underlying distribution of given data. We provide here some suggestions on modeling:

	1. The module uses log link function. We suggest that you start with large penalty scale and/or small learning step. For example, setting alpha to 100 and learning rate to 0.01.

	2. We suggest that you initialize model weights at 0 when learning rate is small.

	3. The current version of HeteroPoisson module does not support multi-host, but it also accepts weight difference as convergence criteria.

	4. The current version does not include over-dispersion term.