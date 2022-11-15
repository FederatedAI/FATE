# Federated Sampling

From Fate v0.2 supports sample method. Sample module supports threee sample
modes: random sample mode, stratified sample mode, and exact sample by weight.

  - In random mode, "downsample" and "upsample" methods are provided.
    Users can set the sample parameter "fractions", which is the sample
    ratio within data.

  - In stratified mode, "downsample" and "upsample" methods are also 
  provided. Users can set the sample parameter "fractions" too, but it
  should be a list of tuples in the form (label\_i, ratio). Tuples in the
  list each specify the sample ratio of corresponding label. e.g.

  - When using `exact_by_weight` mode, samples will be duplicated `ceil(weight)` copies.
  Any zero-weighted samples will be discarded. Note that this mode requires that instances
  have match id: please set `extend_sid` in configuration 
  when [uploading data](../tutorial/pipeline/pipeline_tutorial_uploading_data_with_meta.ipynb) for this sample mode.

> 
> 
>     [(0, 1.5), (1, 2.5), (3, 3.5)]

<!-- mkdocs
## Param

::: federatedml.param.sample_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
 -->
