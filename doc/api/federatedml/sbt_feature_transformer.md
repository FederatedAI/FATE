# SBT Feature Transformer

A feature engineering module that encodes sample using leaf indices
predicted by Hetero SBT/Fast-SBT. Samples will be transformed into
sparse 0-1 vectors after encoding. See [original
paper](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)
for its details.

![Figure 5: Encoding using leaf
indices\</div\>](../../images/gbdt_lr.png)

## Param

::: federatedml.param.sbt_feature_transformer_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
