Stepwise
=========

Stepwise is a simple, effective model selection technique. FATE provides stepwise wrapper for heterogeneous linear models. The compatible models are listed below:

- `Heterogeneous Logistic Regression <../../linear_model/logistic_regression/README.rst>`_

- `Heterogeneous Linear Regression <../../linear_model/linear_regression/README.rst>`_

- `Heterogeneous Poisson Regression <../../linear_model/poisson_regression/README.rst>`_

Please note that due to lack of loss history, Stepwise does not support multi-host modeling. 

Another point to notice is that Stepwise Module currently does not support validation strategy or early stopping. While validate data may be set in job configuration file, the validate data will not be used.

To use stepwise, set 'need_stepwise' to `True` and specify stepwise parameters as desired. Below is an example of stepwise parameter setting in job configuration file.

.. code-block:: json

	{
		"stepwise_param": {
		        "score_name": "AIC",
		        "direction": "both",
		        "need_stepwise": true,
		        "max_step": 3,
		        "nvmin": 2,
		        "nvmax": 6
		    }
		}

 

For examples of using stepwise with linear models, please refer to `examples/federatedml-1.x-examples/hetero_stepwise`. For explanation on each stepwise module parameter, please refer to the comments in stepwise param :download:`stepwise_param.py <../../param/stepwise_param.py>`.

Please note that on FATE Board, the model information (max iters & coefficient/intercept values) represents the final result model. 


Param
------


.. automodule:: federatedml.param.stepwise_param
   :members:
