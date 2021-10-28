Local Baseline
==============

Local Baseline module builds a sklearn model with local data. The module
is basically a wrapper for sklearn model such that it can be run as part
of FATE job workflow and configured as other federatedml modules.

Use
---

Local Baseline currently only supports sklearn Logistic Regression
model.

The module receives train and, if provided, validate data as specified.
Input data sets must be uploaded beforehand as with other federatedml
models. Local Baseline accepts both binary and multi-class data.

sklearn model parameters may be specified in dict form in job config
file. Any parameter unspecified will take the default value set in
sklearn.

Local Baseline has the same output as matching FATE model, and so its
visualization on FATE Board will have matching format. Nonetheless, note
that Local Baseline is run locally, so other parties will not show
respective information like Logistic Regression module. In addition,
note that loss history is not available for Guest when running
homogeneous training.

For examples of using Local Baseline module, please refer
[here](../../../examples/pipeline/local_baseline).

Param
-----

::: {.automodule}
federatedml.param.local\_baseline\_param
:::
