# Sample Weight

Sample Weight assigns weight to input sample. Weight may be specified by
input param `class_weight` or `sample_weight_name`. Output data
instances will each have a weight value, which will be used for
training. While weighted instances may be used for
prediction(SampleWeight component will assign weights to instances if
prediction pipeline includes this component), Evaluation currently does
not take weights into account when calculating metrics.

If result weighted instances include negative weight, a warning message
will be given.

Please note that when weight is not None, only `weight_diff` convergence
check method may be used for training GLM.


:exclamation:

    If both `class_weight` and `sample_weight_name` are provided, values
    from column of `sample_weight_name` will be used.

<!--  mkdocs
## Param

::: federatedml.param.sample_weight_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
 -->
