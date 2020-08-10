Local Baseline
==============

Local Baseline module builds a sklearn model with local data. The module is basically a wrapper for sklearn model such that it may be configured in job config file like other federatedml models.

Use
------

Local Baseline currently only supports sklearn Logistic Regression model. It supports homogeneous (both Guest & Host) and heterogeneous (Guest only) learning cases.

The module receives train and, if provided, validate data as specified in job config file. The data sets must be uploaded beforehand as with other federatedml models.
Local Baseline now accepts both binary and multi-class data.

sklearn model parameters may be specified in dict form in job config file. Any parameter unspecified will take the default value set in sklearn.

Local Baseline has the same output as matching FATE model, and so its visualization on FATE Board will have matching format.
Nonetheless, note that Local Baseline is run locally, so other parties will not show respective information like Logistic Regression module.
In addition, note that loss history is not available for Guest when running homogeneous training.

Currently local baseline does not support predict task from history job. This feature will be added in future updates.

For examples of using Local Baseline module, please refer `examples/federatedml-1.x-examples/local_baseline`.

Param
------

.. automodule:: federatedml.param.local_baseline_param
   :members: