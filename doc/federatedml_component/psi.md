# Population Stability Index (PSI)

## Introduction

Population stability index (PSI) is a metric to measure how much a
feature has shifted in distribution between two sample sets. Usually,
PSI is used to measure the stability of models or qualities of features.
In FATE, PSI module is used to compute PSI values of features between
two tables.

Given two data columns: expect & actual, PSI will be computed by the
following steps: \* expect column and actual column conduct quantile
feature binning \* compute interval percentage, which is given by (bin
sample count)/(total sample number) \* compute PSI value: psi = sum(
(actual\_percentage - expect\_percentage) \* ln(actual\_percentage /
expect\_percentage) )

For more details of psi, you can refer to this
 [\[PSI tutorial\].](https://www.lexjansen.com/wuss/2017/47\_Final\_Paper\_PDF.pdf).


## Param

  - max\_bin\_num: int, max bin number of quantile feature binning
  - need\_run: bool, need to run this module in DSL
  - dense\_missing\_val: int, float, string imputed missing value when
    input format is dense, default is set to np.nan. Default setting is
    suggested

<!-- mkdocs
## Examples

{ % include-examples "psi" %}
-->

