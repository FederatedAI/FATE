### Hetero SecureBoost

Gradient Boosting Decision Tree(GBDT) is a widely used statistic model for classification and regression problems. 
FATE has provided a novel lossless privacy-preserving tree-boosting system known as [SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/abs/1901.08755).

This federated learning system allows a learning process to be jointly conducted over multiple parties with partially 
common user samples but different feature sets, which corresponds to a vertically partitioned virtual data set. An 
advantage of SecureBoost is that it provides the same level of accuracy as the non privacy-preserving approach while 
at the same time, reveal no information of each private data provider.

The following figure shows the proposed Federated SecureBoost framework.
<div style="text-align:center" align=center>
<img src="./images/secureboost.png" alt="framework" width="500" height="250" />
<br/>
Figure 1: Framework of Federated SecureBoost</div>

Active Party:
We define the active party as the data provider who holds both a data matrix and the class label.
Since the class label information is indispensable for supervised learning, there must be an active party with access 
to the label y. The active party naturally takes the responsibility as a dominating server in federated learning.

Passive Party:
We define the data provider which has only a data matrix as a passive party.
Passive parties play the role of clients in the federated learning setting. They are also in need of building a model 
to predict the class label y for their prediction purposes. Thus they must collaborate with the active party to 
build their model to predict y for their future users using their own features.

We align the data samples under an encryption scheme by using the privacy-preserving protocol for inter-database 
intersections to find the common shared users or data samples across the parties without compromising the 
non-shared parts of the user sets.

To ensure security, passive parties cannot get access to gradient and hessian directly. 
We use a "XGBoost" like tree-learning algorithm. In order to keep gradient and hessian confidential, we require the active party to 
encrypt gradient and hessian before sending them to passive parties. 

By following the SecureBoost framework, multiple parties can jointly build tree ensembled model without leaking privacy 
in federated learning. If you want to learn more about the algorithm details, you can read the paper attached above.

#### Optimization in Parallel Learning
SecureBoost use data parallel learning algorithm to build the decision trees in every party.  
The procedure of the data parallel algorithm in each party is:  
* Every party use mapPartitions API interface to generate feature-histograms of each partition of data.
* Use reduce API interface to merge global histograms from all local feature-histograms
* Find the best splits from merged global histograms by federated learning, then perform splits.

#### Applications
SecureBoost supports the following applications.  
* binary classification, the objective function is sigmoid cross-entropy  
* multi classfication, the objective function is softmax cross-entropy
* regression, objective function now support is least-squared-error-loss、least-absolutely-error-loss、huber-loss、
tweedie-loss、fair-loss、 log-cosh-loss

#### Other features
* Column sub-sample
* Allow use max_split_nodes setting to avoid memory limit exceed, by finding splits of at most number of 
max_split_nodes nodes instead of using all nodes of each level of the tree.
* Support feature importance calculation
* Support Multi-host and single guest to build model
* Support different encrypt-mode to balance speed and security
* Support missing value in train and predict process. 
 


