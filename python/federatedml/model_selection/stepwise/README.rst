Stepwise
=========

Stepwise is a simple, effective model selection technique. FATE provides stepwise wrapper for heterogeneous linear models.
The compatible models are listed below:

- `Heterogeneous Logistic Regression <../../linear_model/logistic_regression/README.rst>`_

- `Heterogeneous Linear Regression <../../linear_model/linear_regression/README.rst>`_

- `Heterogeneous Poisson Regression <../../linear_model/poisson_regression/README.rst>`_

Please note that due to lack of loss history, Stepwise does not support multi-host modeling. 

Stepwise Module currently does not support validation strategy or early stopping.
While validate data may be set in job configuration file, it will not be used in the stepwise process.

To use stepwise, set 'need_stepwise' to `True` and specify stepwise parameters as desired.
Below is an example of stepwise parameter setting in job configuration file.

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

 

For examples of using stepwise with linear models, please refer `here <../../../../examples/dsl/v2/hetero_stepwise>`__.
For explanation on stepwise module parameters, please refer to :download:`stepwise_param <../../param/stepwise_param.py>`.

Please note that on FATE Board, shown model information (max iters & coefficient/intercept values) are of the final result model.


Param
------


.. automodule:: federatedml.param.stepwise_param
   :members:
