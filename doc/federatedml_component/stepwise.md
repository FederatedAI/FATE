# Stepwise

Stepwise is a simple, effective model selection technique. FATE provides
stepwise wrapper for heterogeneous linear models. The compatible models
are listed below:

  - [Heterogeneous Logistic Regression](logistic_regression.md)
  - [Heterogeneous Linear Regression](linear_regression.md)
  - [Heterogeneous Poisson Regression](poisson_regression.md)

Please note that due to lack of loss history, Stepwise does not support
multi-host modeling.

Stepwise Module currently does not support validation strategy or early
stopping. While validate data may be set in job configuration file, it
will not be used in the stepwise process.

To use stepwise, set 'need\_stepwise' to
<span class="title-ref">True</span> and specify stepwise parameters as
desired. Below is an example of stepwise parameter setting in job
configuration file.

``` sourceCode json
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
```

For explanation on
stepwise module parameters, please refer to
[stepwise param](../../python/federatedml/param/stepwise_param.py).

Please note that on FATE Board, shown model information (max iters &
coefficient/intercept values) are of the final result model.

<!-- mkdocs
## Param

::: federatedml.param.stepwise_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

<!-- mkdocs
## Examples

{% include-examples "hetero_stepwise" %}
-->
