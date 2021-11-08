# Scorecard

## Introduction

A [credit scorecard](https://en.wikipedia.org/wiki/Credit_scorecards) is
a credit model for measuring individuals' creditworthiness. By
quantifying the probability that a lender may display a defined
behavior, scorecards represents the lender's creditworthiness in numeric
credit score.

Scorecard module of FATE provides a score transformer which scales
predict score(probability of default) to credit score with user-defined
range and parameter values.

<!-- mkdocs
## Param

::: federatedml.param.scorecard_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

## How to Use

  - params

  - method  
    score method, currently only supports "credit"

  - offset  
    score baseline, default 500

  - factor  
    scoring step, when odds double, result score increases by this
    factor, default 20

  - factor\_base  
    factor base, value ln(factor\_base) is used for calculating result
    score, default 2

  - upper\_limit\_ratio  
    upper bound for odds, credit score upper bound is
    upper\_limit\_ratio \* offset, default 3

  - lower\_limit\_value  
    lower bound for result score, default 0

  - need\_run  
    Indicate if this module needs to be run, default True

<!-- mkdocs

## Examples

{% include-examples "scorecard" %}
-->
