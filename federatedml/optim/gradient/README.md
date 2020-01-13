## Linear Model Gradient Introduction

Currently, we support hetero-lr, homo-lr, hetero-linear regression and hetero-poisson regression. In this folder, we use a unified gradient calculation process template for all hetero linear algorithms.

We also provide a quansi-newton method for hetero-lr and hetero-linear regression.

### Stochastic Quansi-Newton

When using Newton method, we use the following equation to update gradients.

<img src="http://latex.codecogs.com/gif.latex?\w_{k+1}=w_k-\alpha_k*H^{-1}\nablaF(w_k) />

where H is Hessian matrix of w.

However, getting Hessian matrix is computational expensive. Thus, a more feasible solution is use quansi-newton methods. We implement a stochastic quansi-newton method whose process can be shown as below.

 <div style="text-align:center", align=center>
<img src="../images/sqn_1.png" alt="samples" width="500" height="300" /><br/>
Figure 1： Stochastic Quasi-Newton Method
</div>

 <div style="text-align:center", align=center>
<img src="../images/sqn_2.png" alt="samples" width="500" height="300" /><br/>
Figure 2： Hessian Updating
</div>

For more details, please refer to this [paper](https://arxiv.org/abs/1912.00513v2)