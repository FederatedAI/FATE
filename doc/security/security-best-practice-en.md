# Fate security best practice guidance

When the FATE is used to perform tasks such as federated learning, security models and security protocols should be used to ensure that no privacy data is disclosed during the calculation process. This draft attempts to provide security instructions for using FATE.
## 1. State of Art of FATE

FATE includes multiple federal learning protocols, some of which are composed of participants and third-party arbiters, such as Homo LR. Other federated learning protocols do not require third-party arbiter to do parameter aggregation, such as secureboost, Hetero Neural Network, Hetero FTL, etc. Therefore, FATE security model can be divided into two types: with third-party arbiter and without third-party arbiter.

For the federated learning without third-party arbiter, its security assumption is: each participant in FATE is an honest, curious and independent security domain, and the data interaction between participants can ensure that the privacy data of each security domain is not leaked. For the federated learning with third-party arbiter, the security assumption is: the federal learning participants are honest, curious and independent security domains, and the arbiter is honest and curious for all participants. That is, the arbiter will comply with the FATE protocol's execution, but may retain or inference aggregation stage intermediate data. When multiple arbiters perform aggregation and arbitration operations, they must not collude with each other.

Security models of FATE need to use security protocols such as secure multi-party computing to protect data security during data interaction stage between different participants. FATE security protocols include Paillier Encryption, RSA Encryption, Hash Factory, DH Key Exchange, SPDZ, OT, etc.

## 2. Network Security Suggestions

For the participating nodes in FATE, client_authentication and permission components need to be used to realize identity authentication function between nodes.

In the network transmission process involved in FATE, such as HTTPS, Flask, gRPC, etc., SSL/TLS protocol is used to protect the security of data transmission. In particular, SSL/TLS is required for network transmission in a public network environment. TLS key strength complies with RFC8446 section9.1.

FATE usually expose one Network port: 9370，network data package of other ports need to be treated with caution.

## 3. Security Protocol Configuration Recommendations

The federated learning security protocol that FATE supports are: Paillier Homomorphic Encryption, Hash Factory, DH Key Exchange, SecretShare MPC Protocol, Oblivious Transfer. The table below shows the security configuration recommendation. 

| Protocol Name        | Algorithm Classification   | Security Configuration Recommendation |
| :-------------:| :----------: | :------------: |
| Paillier Encryption|asymmetric encryption algorithm   | Refer to RSA algorithm requirement, minimum key length (module length) is 1024 bits, 2048 bits is recommended |
| RSA Encryption     |asymmetric encryption algorithm   |minimum key length (module length) is 1024 bits, 2048 bits is recommended  |
| ECDH Encryption    | asymmetric encryption algorithm based on elliptic curve |        key length 256bits   |
| Diffie Hellman Key Exchange|key exchange algorithm| minimum key length 1024bits, 2048bits recommended      |
| Hash Factory       |hash algorithm        |sha256，sm3 are recommended            |
| SecretShare MPC Protocol (SPDZ) | secret share algorithm  | participants in secret share cannot collusion with each other, and two participants at least; use trusted third party or Paillier Encryption to generate multiply triples |
|Oblivious Transfer| based on RSA encryption algorithm    | minimum key length is 1024 bits, 2048bits is recommended    |

## 4. FederatedML Components Security Recommendations

When the security assumptions of FATE participating nodes are honest and curious, adversary cannot obtain valid plaintext information about the model and intermediate gradient data under the protection of security protocol described in Section 3. For example, model inference, reconstruction and GAN attacks require black or white box access to the model of participating nodes, or access to plaintext gradient data. When these data are protected by homomorphic encryption, secret sharing and other security protocols, the attack is not easy to implement in  FATE security assumption.

When a participant or arbiter is compromised by system attack, or the participant itself is a malicious node, it may expose local model, intermediate data and overall model. In view of this hypothesis, TEE can be adopted to protect the operational environment of FATE participants. Or using differential privacy to protect the confidentiality of participants' private data, to make sure inference, reconstruction, free-riding attacks can not obtain valid private data. Or using non-technical methods, like background check, signing agreement, etc., to ensure that all participants are at least semi-honest security model. According to this compromise hypothesis, data poisoning and model poisoning may also exist. These attacks only affect model availability, but does not reveal private data. When FATE participants find that machine learning model cannot converge or the classification accuracy decreases, the learning process should be terminated in time.

The following table shows the security models and related security protocol recommendations required by different federated learning components. Among them, security model mainly refers to whether the third party node is needed as arbiter, and the role of each participant. Existing security mechanisms or security recommendations refer to security mechanisms adopted by the federated learning algorithm or recommended security protocols and other security suggestions.

| Components | Algorithm Description      |       Security Model  | Exist Security Mechanism or Security Recommendations |
| :-------------:| :-------------: | :-------------: |:-------------:|
|  Intersect  |  This module helps two and more parties to find common entry ids without leaking non-overlapping ids. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/intersect/)            |    No third party arbiter is needed.           |  RSA, DH, ECDH encryption algorithms are used to protect data privacy between participants, and the ID information in the intersection will be known by participants.  
| Hetero Feature Binning  | Based on quantile binning and bucket binning methods, Guest (with label) could evaluate Host's data binning quality with WOE/IV/KS. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_binning/) | No third party arbiter is needed | Participants use Paillier Encryption to protect label information|
|Hetero Feature Selection| If iv filter is used, Hetero Feature Binning will be invoked. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_selection/)|No third party arbiter is needed. |Iv filter's security recommendations refer to Hetero Feature Binning. |
|Hetero LR|Hetero logistic regression between host and guest.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-lr)|Third party arbiter is needed, Party A holds the label. |Arbiter needs to use Paillier Encryption and mask to protect gradients.|
|Hetero SSHELR|heterogeneous logistic regression without arbiter role.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-sshe-logistic-regression) |No third parity arbiter is needed| Use Secure Matrix Multiplication Protocol, in which uses Paillier and Secret Sharing |
|Homo LR|Homogeneous logistic regression with arbiter role [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#homogeneous-lr)|Participants are identical, and third party arbiter is needed| Secure aggregation uses Secret Sharing Protocol, and t>n/2 (t is the minimum pieces of rebuilding secret key, n is total number of secret sharing pieces). In order to prevent server from making up fake clients，PKI could be used to provide registration for clients. Use double-masking to prevent server from obtaining client gradient data when processing offline or network delay clients.|
|Hetero LinR|Heterogeneous linear regression.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/linear_regression/#heterogeneous-linr)|Party A represents guest and has the label, Party B represents Host, Party C is third party arbiter. | Gradients aggregation uses Paillier Encryption protocol or Secret Sharing protocol. If using Paillier Encryption, additional mask should be used to prevent arbiter from obtaining gradient data. If using Secret Sharing, at least two secret sharing servers are needed, and servers cannot collusion.|
|HeteroPoisson |Heterogeneous Federated Poisson Regression [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/poisson_regression/)|Party A represents guest and has the label, Party B represents Host, Party C is third party arbiter. | Gradients aggregation uses Paillier Encryption protocol or Secret Sharing protocol. If using Paillier Encryption, additional mask should be used to prevent arbiter from obtaining gradient data. If using Secret Sharing, at least two secret sharing servers are needed, and servers cannot collusion.|
|HomoNN|Homogeneous Neural Network [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/homo_nn/)|Participants are identical, third party arbiter is needed|Secure aggregation uses Secret Sharing Protocol, and t>n/2 (t is the minimum pieces of rebuilding secret key, n is total number of secret sharing pieces). In order to prevent server from making up fake clients，PKI could be used to provide registration for clients. Use double-masking to prevent server from obtaining client gradient data when processing offline or network delay clients.|
|Hetero Secure Boosting|Heterogeneous secure gradient boosting decision tree. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#hetero-secureboost )| Active Party is the dominating server and holds the label y. Passive Party only provide data matrix. |Active party generate Paillier Encryption key pair to protect intermediate data.|
|Hetero NN| Heterogeneous neural network. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_nn/)|Party A only provides data matrix. Party B is dominating server, provides data matrix and label y.|Party B generates Paillier Encryption key pair to protect activations. Party A and B add noise to activations to avoid inference.|
|Homo Secure Boost|Homogeneous secure gradient boosting decision tree. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#homo-secureboost)|Participants(clients) are identical, third party arbiter(server) is needed.| Clients negotiate random number which could be attached to parameter G anda H. This random number could protect intermediate data from being known by server.|
|Hetero FTL|Heterogeneous federated transfer learning. [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_ftl/)|Host party owns data matrix. Guest party owns data matrix and label. No third party arbiter is needed. |Host and Guest uses Paillier Encryption and mask to protect intermediate data.|
|Hetero KMeans| Heterogeneous KMeans[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_kmeans/)|Party A represents as Guest, Party B represents as Host. Third party arbiter is needed, and is in charge of generating private and public keys.| Guest and Host negotiate random number and attach it to intermediate data. This random number could protect intermediate data from being known by arbiter. |



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








