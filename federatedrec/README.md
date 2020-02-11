### Federated Recommondation

#### 1. Introduction

Federatedrec aims to address recommondation problems such as rating prediction and item ranking under federated learning senario. It includes implementation of a number of popular recommondation algorithms based on FATE library. Such as federated fm, federated mf, federated svd etc.

#### 2. Background

With laws related to the protection of data security and privacy, such as General Data Protection Regulation (GDPR), coming out in recent years, data collection becomes more difficult. And users give more attention to the problem of data privacy. Directly sharing user data between companies (organizaiotns) is undesired. Such data silo issues commonly exist in recommender systems.

FedRec addresses the data silo issue and builds centralized recommender without compromising privacy and security. FedRecLib includes implementation of a suite of state-of-the-art recommondation algorithms based on FATE library.

#### 3. Algorithms list:

##### 1. [Hetero FM(factorization machine)](./factorization_machine/README.md)
Build hetero factorization machine module through multiple parties.

- Corresponding module name: HeteroFM
- Data Input: Input DTable.
- Model Output: Factorization Machine model.


##### 2. [Hetero MF(matrix factorization)](./matrix_factorization/README.md)
Build hetero matrix factorization module through multiple parties.

- Corresponding module name: HeteroMF
- Data Input: Input DTable of user-item rating matrix data.
- Model Output: Matrix Factorization model.

More available algorithms are coming soon.
