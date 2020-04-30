### Federated Recommondation

#### 1. Introduction

Federatedrec aims to address recommondation problems such as rating prediction and item ranking under federated learning scenario. It includes implementation of a number of popular recommondation algorithms based on FATE library. Such as federated fm, federated mf, federated svd etc.

#### 2. Background

With laws related to the protection of data security and privacy, such as General Data Protection Regulation (GDPR), coming out in recent years, data collection becomes more difficult. And users give more attention to the problem of data privacy. Directly sharing user data between companies (organizaiotns) is undesired. Such data silo issues commonly exist in recommender systems.

FedRec addresses the data silo issue and builds centralized recommenders without compromising privacy and security. FedRecLib includes implementations of a suite of state-of-the-art recommendation algorithms based on FATE library.

#### 3. Algorithms list:

##### 1. [Hetero FM(factorization machine)](./factorization_machine/README.md)
Build a hetero factorization machine model through multiple parties.

- Corresponding module name: HeteroFM
- Data Input: Input DTable.
- Model Output: Factorization Machine model.

##### 2. [Homo-FM](./factorization_machine/README.md)
Build a homo factorization machine model through multiple parties.

- Corresponding module name: HomoFM
- Data Input: Input DTable.
- Model Output: Factorization Machine model.

##### 3. [Hetero MF(matrix factorization)](./matrix_factorization/README.md)
Build a hetero matrix factorization model through multiple parties.

- Corresponding module name: HeteroMF
- Data Input: Input DTable of user-item rating matrix data.
- Model Output: Matrix Factorization model.


##### 4. [Hetero SVD](./svd/README.md)
Build a hetero SVD model through multiple parties.

- Corresponding module name: HeteroSVD
- Data Input: Input DTable of user-item rating matrix data.
- Model Output: Hetero SVD model.


##### 5. [Hetero SVD++](./svdpp/README.md)
Build a hetero SVD++ model through multiple parties.

- Corresponding module name: HeteroSVDPP
- Data Input: Input DTable of user-item rating matrix data.
- Model Output: Hetero SVD++ model.

##### 6. [Hetero GMF](./general_mf/README.md)
Build a hetero GMF model through multiple parties.

- Corresponding module name: HeteroGMF
- Data Input: Input DTable of user-item rating matrix data(using positive data only).
- Model Output: Hetero GMF model.

More available algorithms are coming soon.
