# FATE security best practice recommendations

When FATE is used to perform tasks such as federated learning, security models and security protocols should be used to ensure that no private data is disclosed during the whole process. This document attempts to provide security recommendations for FATE users.

## 1. State of Art of FATE

FATE includes multiple federated learning protocols, some of which are composed of participants and third-party Arbiter, such as Homo LR, etc. Other federated learning protocols do not require third-party Arbiter to do parameter aggregation, such as secureboost, Hetero Neural Network, Hetero FTL, etc. All the federated learning protocols need to use security protocols such as secure multi-party computing to protect data security during data interaction stage between different participants. FATE security protocols include Paillier Encryption, RSA Encryption, Hash Factory, DH Key Exchange, SPDZ, OT, etc. In terms of security, the threat model of FATE could be described as follow：
 - When each participant in FATE is an independent, honest-but-curious security domain, data interaction between participants can ensure that private data of each participants will not be leaked. Adversary cannot obtain valid plaintext information about the model and intermediate gradient data under the protection of security protocol. For example, model inference, reconstruction and GAN attacks require black or white box access to the model of participating nodes, or access to plaintext gradient data. When these data are protected by homomorphic encryption, secret sharing and other security protocols, these attacks are not easy to implement in FATE.
 - When a participant or Arbiter is compromised by system attack, or the participant itself is a malicious node, it may expose intermediate data, local model or full model, which made inference, reconstruction, free-riding attacks possible. According to this compromise hypothesis, data poisoning and model poisoning may also exist. These attacks only affect model availability, but does not reveal private data. When FATE participants find that machine learning model cannot converge or the classification accuracy decreases or other abnormal situations, the learning process should be terminated in time. In terms of this hypothesis, TEE can be adopted to protect the processing environment of FATE participants. Or using differential privacy to protect the confidentiality of participants' private data, to make sure attacker cannot obtain accurate private data. Or using non-technical methods, like background check, signing agreement to ensure that all participants must comply with honest-but-curious security model. 

## 2. Network Security Suggestions

For the participating nodes in FATE, client_authentication and permission components need to be used to execute identity authentication function between nodes.

In the network transmission process in FATE, protocols like HTTPS, Flask, gRPC could use TLS to protect data-in-transit. In particular, TLS is mandantory requirement for network transmission in a public network environment. TLS key strength complies with RFC8446 section9.1.

When using Eggroll as computing engine and Rollsite as transport engine, Rollsite exposes port 9370. When using Spark as computing engine, Pulsar or RabbitMQ as transport engine, Pulsar exposes port 6650 or 6651, RabbitMQ exposes port 5672, and Nginx exposes port 9300 and 9310. Network data package of other ports should be treated with caution.

## 3. Security Protocol Configuration Recommendations

The federated learning security protocol that FATE supports are: Paillier Homomorphic Encryption, Hash Factory, DH Key Exchange, SecretShare MPC Protocol, Oblivious Transfer. The table below shows the security configuration recommendation. 

| Protocol Name        | Algorithm Classification   | Security Configuration Recommendation |
| :-------------:| :----------: | :------------: |
| Paillier Encryption|asymmetric encryption algorithm   | Refer to RSA algorithm requirement, minimum key length (module length) is 1024 bits, 2048 bits is recommended |
| RSA Encryption     |asymmetric encryption algorithm   |minimum key length (module length) is 1024 bits, 2048 bits is recommended  |
| ECDH Encryption    | asymmetric encryption algorithm based on elliptic curve |        key length 256bits   |
| Diffie Hellman Key Exchange|key exchange algorithm| minimum key length 1024bits, 2048bits recommended      |
| Hash Factory       |hash algorithm        |sha256，sm3 are recommended            |
| SecretShare MPC Protocol (SPDZ) | secret share algorithm  | participants in secret share cannot collude with each other, and two participants at least; use trusted third party or Paillier Encryption to generate multiply triples |
|Oblivious Transfer| based on RSA encryption algorithm    | minimum key length is 1024 bits, 2048bits is recommended    |

## 4. FederatedML Components Security Recommendations

The following table shows the security models and related security protocol recommendations required by different federated learning components. Among them, security model mainly refers to whether the third party node is needed as Arbiter, and the role of each participant. Existing security mechanisms or security recommendations refer to security mechanisms adopted by the federated learning algorithm or recommended security protocols and other security suggestions.

| Components | Algorithm Description      |       Security Model  | Exist Security Mechanism or Security Recommendations |
| :-------------:| :-------------: | :-------------: |:-------------:|
|  Intersect  |  This module helps two parties to find common entry ids without leaking non-overlapping ids. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/intersect/)            |    No third party Arbiter is needed.           |  RSA, DH, ECDH encryption algorithms are used to protect data privacy between participants, and the ID information in the intersection will be known by participants.  
| Hetero Feature Binning  | Based on quantile binning and bucket binning methods, Guest (with label) could evaluate Host's data binning quality with WOE/IV/KS. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_binning/) | No third party Arbiter is needed | Participants use Paillier Encryption to protect label information|
|Hetero Feature Selection| If iv filter is used, Hetero Feature Binning will be invoked. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_selection/)|No third party Arbiter is needed. |Iv filter's security recommendations refer to Hetero Feature Binning. |
|Hetero LR|Hetero logistic regression between Host and Guest.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-lr)|Third party Arbiter is needed, Guest(Party A) holds the label. |Arbiter needs to use Paillier Encryption and mask to protect gradients.|
|Hetero SSHELR|Heterogeneous logistic regression without Arbiter role.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-sshe-logistic-regression) |No third parity Arbiter is needed| Use Secure Matrix Multiplication Protocol, in which uses Paillier and Secret Sharing |
|Homo LR|Homogeneous logistic regression with third party Arbiter role [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#homogeneous-lr)|Participants are identical, and third party Arbiter is needed| Secure aggregation uses Secret Sharing Protocol, and t>n/2 (t is the minimum pieces of rebuilding secret key, n is total number of secret sharing pieces). In order to prevent Arbiter from making up fake Guest/Host, PKI could be used to provide registration for participants. Use double-masking to prevent Arbiter from obtaining gradient data when processing offline or network delay participants.|
|Hetero LinR|Heterogeneous linear regression.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/linear_regression/#heterogeneous-linr)|Guest(Party A) has the label, Party B represents Host, Party C is third party Arbiter. | Gradients aggregation uses Paillier Encryption protocol or Secret Sharing protocol. If using Paillier Encryption, additional mask should be used to prevent Arbiter from obtaining gradient data. If using Secret Sharing, at least two secret sharing Arbiters are needed, and Arbiters cannot collusion.|
|HeteroPoisson |Heterogeneous Federated Poisson Regression [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/poisson_regression/)|Party A represents Guest and has the label, Party B represents Host, Party C is third party Arbiter. | Gradients aggregation uses Paillier Encryption protocol or Secret Sharing protocol. If using Paillier Encryption, additional mask should be used to prevent Arbiter from obtaining gradient data. If using Secret Sharing, at least two secret sharing Arbiters are needed, and Arbiters cannot collusion.|
|HomoNN|Homogeneous Neural Network [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/homo_nn/)|Participants are identical, third party Arbiter is needed|Secure aggregation uses Secret Sharing Protocol, and t>n/2 (t is the minimum pieces of rebuilding secret key, n is total number of secret sharing pieces). In order to prevent Arbiter from making up fake participants, PKI could be used to provide registration for participants. Use double-masking to prevent Arbiter from obtaining gradient data when processing offline or network delay participants.|
|Hetero Secure Boosting|Heterogeneous secure gradient boosting decision tree. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#hetero-secureboost )| Active Party is the dominating server and holds the label y. Passive Party only provide data matrix. |Active party generate Paillier Encryption key pair to protect intermediate data.|
|Hetero NN| Heterogeneous neural network. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_nn/)|Party A only provides data matrix. Party B is dominating server, provides data matrix and label y.|Party B generates Paillier Encryption key pair to protect activations. Party A and B add noise to activations to avoid inference.|
|Homo Secure Boost|Homogeneous secure gradient boosting decision tree. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#homo-secureboost)|Participants are identical, third party Arbiter is needed.| Participants negotiate random number which could be attached to parameter G anda H. This random number could protect intermediate data from being known by Arbiter.|
|Hetero FTL|Heterogeneous federated transfer learning. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_ftl/)|Host party owns data matrix. Guest party owns data matrix and label. No third party Arbiter is needed. |Host and Guest uses Paillier Encryption and mask to protect intermediate data.|
|Hetero KMeans| Heterogeneous KMeans[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_kmeans/)|Party A represents as Guest and holds label, Party B represents as Host. Third party Arbiter is needed, and is in charge of generating private and public keys.| Guest and Host negotiate random number and attach it to intermediate data. This random number could protect intermediate data from being known by Arbiter. |



## 5. Compliance Recommendations
FATE is an open source federated machine learning platform. All participants exchange intermediate data through security protocols to make sure their own data stays within domain, and achieve the purpose of jointly training machine learning model in the end. When FATE is used in practices, some laws and regulations related to information security, network security and data security could help users to use FATE more effectively from the perspective of security and compliance. The following lists some domestic and international laws and regulations that may be involved in the process of using FATE.

### Relevant domestic laws and regulations
 - Data Security Law of the People's Republic of China
 - Personal Information Protection Law of the People's Republic of China
 - Network Security Law of the People's Republic of China
 - PIA GB∕T 39335-2020 Information Security Technology  Personal Information Security Impact Assessment Guide

### Relevant international laws and regulations

 - GDPR-General Data Protection Regulation
 - CCPA-California Consumer Privacy Act Regulation
 - HIPAA-Health Insurance Portability and Accountability Act of 1996
 - DPIA-Data Protection Impact Assessment
