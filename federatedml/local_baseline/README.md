### Local Baseline

Local Baseline module builds a sklearn model with local data on Guest's side. The module is basically a wrapper for sklearn model so that it may be configured in job config file like other federatedml models.

### Use

Local Baseline currently only supports sklearn Logistic Regression model.

The module receives train and validate data as specified in job config file. The data sets must be uploaded beforehand as with other federatedml models. sklearn model parameters may be specified in dict form in job config file. Any parameter unspecified will take the default value set in sklearn package.

Local Baseline has the same output as matching FATE model, and so the visualization on FATE Board will have the same format.

For examples of using Local Baseline module, please refer [here](../../examples/federatedml-1.x-examples/local_baseline).