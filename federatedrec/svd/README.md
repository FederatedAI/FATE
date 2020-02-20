# Federated Singular Value Decomposition

SVD is a  supervised learning approach that decompose a matrix into a product of matrices. It does matrix factorization via following formula:

​								$r_{ui}=\mu+b_u+b_i+p_uq_i$

Different to MF, SVD incoporate bias of users as $b_u$ and bias of items as $b_i$.

SVD is commonly used in recommendation senario to decompose a user-item rating matrix into user profile and item profile, and to predict unknown user-item pair's rating by compute the dot product of user profile and item profile.

SVD's loss function:

​								$\mathcal{L}=\sum\limits_{(u,i)\in{\mathbb{K}}}(r_{ui}-\hat{r}_{ui})^2+\lambda(b_u^2+b_i^2+\Vert{q_i}\Vert^2+\Vert{p_u}\Vert^2)$

where λ is a small positive values to rescale the penalizer. Denote $e_{ui}=r_{ui}-{\hat{r}}_{ui}$, and the gradients on different parameters are:

​								$\frac{\partial\mathcal{L}}{\partial{b_u}}=-2\sum\limits_{i\in\mathbb{k}(u)}e_{ui}+2\lambda{b_u}$

​								$\frac{\partial\mathcal{L}}{\partial{b_i}}=-2\sum\limits_{u\in\mathbb{k}(i)}e_{ui}+2\lambda{b_i}$

​								$\frac{\partial\mathcal{L}}{\partial{p_u}}=-2\sum\limits_{i\in\mathbb{k}(u)}e_{ui}q_i+2\lambda{p_u}$

​								$\frac{\partial\mathcal{L}}{\partial{q_i}}=-2\sum\limits_{u\in\mathbb{k}(i)}e_{ui}p_u+2\lambda{q_i}$

## Heterogeneous SVD

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible coordinate training process and encrypted data exchange.

Inspired by VFedMF, we can divide the parameters of SVD into item-related (e.g. $p$) and user-related (e.g. $q$) ones. Based on the setting of same-user and different-item, we let party A, B share the same user-profile and hold item-profile individually. The rating data is protected by keeping the item-profile unknown to each other.

|               | User-Related   | Item-Related |
| ------------- | -------------- | ------------ |
| Parameters    | $b_u,p_u$      | $b_i,p_i$    |
| Training mode | Jointly update | Local update |
|               |                |              |

**User-related parameters:**

​												$\frac{\partial\mathcal{L}}{\partial{b_u}}=-2\sum\limits_{i\in\mathbb{k}_A(u)}e_{ui}-2\sum\limits_{i\in\mathbb{k}_B(u)}e_{ui}+2\lambda{b_u}$

​												$\frac{\partial\mathcal{L}}{\partial{p_u}}=-2\sum\limits_{i\in\mathbb{k}_A(u)}e_{ui}q_i-2\sum\limits_{i\in\mathbb{k}_B(u)}e_{ui}q_i+2\lambda{p_u}$

**Let:**

<div style="text-align:center", align=center>
<img src="./images/fig4.png" alt="samples" width="216" height="219" /><br/>
</div>

Then the parameter updates of user-related parameters can be represented as:

<div style="text-align:center", align=center>
<img src="./images/fig5.png" alt="samples" width="236" height="99" /><br/>
</div>

**Item-related parameters** 

The item-related parameters can be updated locally by A,B using the same equation as regular SVD++.

**Compute** $\mu$

According to equation, we need to compute µ before the training of SVD++, where µ is the global average rating score. Intutively, µ can be computed using following equation.

<div style="text-align:center", align=center>
<img src="./images/fig6.png" alt="samples" width="443" height="60" /><br/>
</div>

## Features:
1. L1 & L2 regularization
2. Mini-batch mechanism
3. Five optimization methods:
    a)	“sgd”: gradient descent with arbitrary batch size
    b) “rmsprop”: RMSProp
    c) “adam”: Adam
    d) “adagrad”: AdaGrad
    e) “nesterov_momentum_sgd”: Nesterov Momentum
4. Three converge criteria:
    a) "diff": Use difference of loss between two iterations, not available for multi-host training
    b) "abs": Use the absolute value of loss
    c) "weight_diff": Use difference of model weights
6. Support validation for every arbitrary iterations
7. Learning rate decay mechanism.