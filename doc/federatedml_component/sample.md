# Federated Sampling

From Fate v0.2 supports sample method. Sample module supports two sample
modes: random sample mode and stratified sample mode.

  - In random mode, "downsample" and "upsample" methods are provided.
    Users can set the sample parameter "fractions", which is the sample
    ratio within data.

\- In stratified mode, "downsample" and "upsample" methods are also
provided. Users can set the sample parameter "fractions" too, but it
should be a list of tuples in the form (label\_i, ratio). Tuples in the
list each specify the sample ratio of corresponding label. e.g.

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
