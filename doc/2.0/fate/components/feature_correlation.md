# Feature Correlation

Feature correlation provides computation of Pearson correlation matrix for local and hetero-federated scenario.
To switch between the two modes, set `local_only` to `True` or `False` accordingly.

Pearson Correlation Coefficient is a measure of the linear correlation between two variables, $X$ and $Y$, defined as,

$$\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X\sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y} = E\left[\left(\frac{X-\mu_X}{\sigma_X}\cdot\frac{Y-\mu_Y}{\sigma_Y}\right)\right]$$

Let

$$\tilde{X} = \frac{X-\mu_X}{\sigma_X}, \tilde{Y}=\frac{Y-\mu_Y}{\sigma_Y}$$

then,

$$\rho_{X, Y} = E[\tilde{X}\tilde{Y}]$$

## Implementation Detail

We use an MPC protocol called SPDZ for Heterogeneous Pearson Correlation
Coefficient calculation. SPDZ([Ivan Damg˚ard](https://eprint.iacr.org/2011/535.pdf),
[Marcel Keller](https://eprint.iacr.org/2017/1230.pdf)) is a
multiparty computation scheme based on somewhat homomorphic encryption
(SHE).


## Features

- local Pearson correlation efficient computation
- hetero-federated Pearson correlation efficient computation
- local VIF computation
- computation on select cols only(use `skip_col`)


