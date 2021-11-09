# Heterogeneous Pearson Correlation Coefficient

## Introduction

Pearson Correlation Coefficient is a measure of the linear correlation between two variables, $X$ and $Y$, defined as,

$$\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X\sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y} = E\left[\left(\frac{X-\mu_X}{\sigma_X}\cdot\frac{Y-\mu_Y}{\sigma_Y}\right)\right]$$

Let

$$\tilde{X} = \frac{X-\mu_X}{\sigma_X}, \tilde{Y}=\frac{Y-\mu_Y}{\sigma_Y}$$

then,

$$\rho_{X, Y} = E[\tilde{X}\tilde{Y}]$$

## Implementation Detail

We use an MPC protocol called SPDZ for Heterogeneous Pearson Correlation
Coefficient calculation. For more details, one can refer [[here](secureprotol.md)]

<!-- mkdocs
## Param

::: federatedml.param.pearson_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

## How to Use

  - params

  - column\_indexes  
    \-1 or list of int. If -1 provided, all columns are used for
    calculation. If a list of int provided, columns with given indexes
    are used for calculation.

  - column\_names  
    names of columns use for calculation.

!!! Note

    if both params are provided, the union of columns indicated are used for calculation.

<!-- mkdocs
## Examples

{% include-examples "hetero_pearson" %}
-->
