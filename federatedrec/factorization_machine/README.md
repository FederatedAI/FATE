# Federated Factorization Machine

Factorization Machine(FM) is a  supervised learning approach incorporates second-order feature interaction.
Federated factorization machine computes these cross-party cross-features and their gradients under encryption. 

Here we simplify participants of the federation process into three parties. Party A represents Guest, party B represents Host. Party C, which is also known as “Arbiter,” is a third party that works as coordinator. Party C is responsible for generating private and public keys.

## Heterogeneous FM

The inference process of HeteroFM is shown below:

<div style="text-align:center", align=center>
<img src="./images/HeteroFM.png" alt="samples" width="500" height="300" /><br/>
Figure 1： Federated HeteroFM</div>

Similar to other hetero federated learning approch, a sample alignment process is conducted before training. The sample alignment process identifies overlapping samples in databases of all parties. The federated model is built based on the overlapping samples. The whole sample alignment process is conducted in encryption mode, and so confidential information (e.g. sample ids) will not be leaked.

In the training process, party A and party B each compute their own linear and cross-features forward results, and compute sucure cross-party cross-features under homomorphic encryption. Arbiter then aggregates, calculates, and transfers back the final gradients to corresponding parties. 

FM prediction over two parties as:
$$
\begin{split}
f([X_p^{(A)};X_q^{(B)}]) {} &=  f(X_p^{(A)})+f(X_q^{(B)})+\sum\limits_{i,j}x_p,i^{(A)}x_q,j^{(B)}  {}\\
	&=f(X_p^{(A)})+f(X_q^{(B)})+\sum_{i}\sum_{j}\sum_{k=1}^{d'}v_{i,k}^{(A)}v_{j,k}^{(B)}x_{p,i}^{(A)}x_{q,j}^{(B)} {}\\
	&=f(X_p^{(A)})+f(X_q^{(B)})+\sum_{k=1}^{d'}(\sum_iv_{i,k}^{(A)}x_{p,i}^{(A)})(\sum_jv_{j,k}^{(B)}x_{q,j}^{(B)}) {}\\
\end{split}
$$
FM loss function over two parties is defined as:
$$
\begin{split}
{} &\ell([W^{(A)};W^{(B)}],[V^{(A)};V^{(B)}])  {}\\
	&=\frac{1}{2nA}\sum_{p=1}^{nA}(y_p-f([X_{p}^{(A)};X_{q}^{(B)}]))^2+\frac{\alpha}{2}\Omega([W^{(A)};W^{(B)}],[V^{(A)};V^{(B)}]) {}\\

\end{split}
$$
where $\alpha>0$ is a hyper-parameter.



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