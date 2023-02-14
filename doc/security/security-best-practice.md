# Fate安全best practice guidance
当使用FATE系统执行联邦学习等任务时，需使用恰当的安全模型和安全协议以确保计算过程中不泄露隐私数据。本文稿尝试给出使用FATE的安全指导。
## 一. FATE 安全现状
FATE包括了多种联邦学习协议，一些联邦学习协议例如Homo LR等由参与方和第三方arbiter组成。另一些联邦学习协议例如secureboost、Hetero Neural Network、Hetero FTL等不需要第三方arbiter做参数聚合。因此从安全角度，FATE安全模型主要分为两种：有第三方arbiter和无第三方arbiter。
对于没有第三方arbiter的联邦学习，其安全假设为：各参与方为诚实且好奇的独立安全域，各参与方的数据交互能够保证各安全域的隐私数据不泄露。对于有第三方arbiter的联邦学习安全假设为：联邦学习各参与方为诚实且好奇的独立的安全域，arbiter对于各参与方为诚实且好奇的安全域，即arbiter会遵守FATE协议执行，但是有可能会保留或推理模型聚合阶段的中间数据。对于多个arbiter共同执行聚合、arbitration等操作时，要求arbiter之间不能共谋。
以上两种安全模型都需要采用安全多方计算等安全协议来保护不同参与方之间数据交互阶段的数据安全。FATE安全协议有Paillier Encryption，RSA Encryption，Hash Factory，DH Key Exchange，SPDZ，OT等。


## 二. 网络安全建议
对于参与计算的FATE节点，各节点之间首先采用client_authentication和permission实现节点的身份认证。

在FATE涉及的网络传输过程中，例如基于HTTPS、Flask、gRPC协议等，采用SSL/TLS协议能够更好的保护数据传输的安全。尤其是在公网环境下进行网络传输时，SSL/TLS是必须选项。TLS密钥强度遵循RFC8446 section9.1。

FATE平台通常暴露一个网络端口：9370， 对于其余网络端口数据包，应慎重处理。

## 三．安全协议配置建议
FATE目前支持的联邦学习安全协议为：paillier同态加密，RSA Encryption，Hash Factory，Diffne Hellman Key Exchange，SecretShare MPC Protocol，Oblivious Transfer。现就相关安全协议给出安全配置建议,如下表所示。


| 协议名称        | 算法分类   | 安全配置建议 |
| :-------------:| :----------: | :------------: |
| Paillier Encryption|非对称加密算法        | 参考RSA算法安全性要求，密钥长度（模长）最低1024位，推荐2048位          |
| RSA Encryption     |非对称加密算法        |密钥长度（模长）最低1024位，推荐2048位           |
| ECDH Encryption    | 基于椭圆曲线的非对称加密算法       |                 密钥长度256位                         |
| Diffie Hellman Key Exchange|密钥交换算法 |密钥长度（模长）最低1024位，推荐2048位          |
| Hash Factory       |散列算法        |推荐使用sha256，sm3           |
| SecretShare MPC Protocol (SPDZ) | 秘密分享算法  | 秘密分享参与方之间不能共谋，并至少需要两个参与方；使用可信第三方生成乘法三元组或者使用paillier Encryption生成乘法三元组|
|Oblivious Transfer| 非对称加密算法         | 密钥长度最低1024位，推荐2048位          |

## 四．FederatedML Components安全建议

当FATE参与节点的安全假设为诚实且好奇时，在第三节所述安全协议的保护下，攻击者无法获得有关模型、中间梯度数据的有效明文信息。例如模型推理、重构以及GAN攻击等，这些攻击方式都要求对参与节点的模型有黑盒或者白盒访问权限，或者能够访问明文梯度数据。当这些数据由同态加密、秘密共享等安全协议保护时，该攻击在FATE安全假设中不易实现。
当某一参与方或者arbiter遭受系统攻击妥协后，其可能暴露局部模型、中间数据和总体模型，这种情况多存在于参与方安全等级较低的场景中。针对此假设，在采用安全协议的基础上可采用TEE保护FATE的运算环境安全；或采用差分隐私保护自身数据的机密性，使推理、重构、搭便车等攻击方式无法获得自身的隐私数据；或采用非技术手段，通过背景调查、签约协议等方式确保各参与方至少为半诚实安全模型。针对此妥协假设也可能发生数据投毒和模型投毒，这种攻击方式只破坏模型可用性，不泄露隐私数据。当使用者发现模型无法收敛，分类精度下降等异常情况时，应及时终止学习过程。
如下表给出了不同的联邦学习components需要的的安全模型以及相关的安全协议建议。其中，安全模型主要指是否需要第三方节点作为arbiter，以及各参与方的角色。既有安全机制或安全建议主要指联邦学习算法已采用的安全机制或推荐采用的安全协议及其它安全建议。

| Components名称 | 算法描述      |       安全模型  | 既有安全机制或安全建议 |
| :-------------:| :----------: | :------------: |:-------:|
|  Intersect  |  计算两方的相交数据集，而不会泄漏任何差异数据集的信息。主要用于纵向任务。[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/intersect/)            |    不需要第三方arbiter            |  参与方之间利用RSA、DH、ECDH加密算法保护数据隐私，交集内ID信息会被共同参与方知晓       |
| Hetero Feature Binning  |基于quantile binning 和 bucket binning 方法, Guest (具有标签) 可以通过WOE/IV/KS评估Host的数据分箱质量。[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_binning/) |不需要第三方arbiter | 参与方之间利用paillier加性同态加密进行数据计算|
|Hetero Feature Selection| 若使用iv fiter，需采用Hetero Feature Binning方法。[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/feature_selection/)|不需要第三方arbiter|iv filter参考Hetero feature binning安全建议 |
|Hetero LR|通过多方构建纵向逻辑回归模块。[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-lr)|需要第三方arbiter，参与方A拥有标签 |Arbiter需利用paillier同态加密和掩码保护梯度数据隐私。|
|Hetero SSHELR|heterogeneous logistic regression without arbiter role.[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#heterogeneous-sshe-logistic-regression) |不需要第三方arbiter|，采用Secure Matrix Multiplication Protocol which uses HE and Secret Sharing|
|Homo LR|横向逻辑回归 [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/logistic_regression/#homogeneous-lr)|参与方身份对等，需要第三方arbiter| 安全聚合采用秘密分享协议，并且t>n/2（t是指恢复密码的最小份数，n为client总数？）；为防止server虚构client数量，推荐使用PKI为client提供注册信息；采用double-mask防止server在处理掉线client或者网络延迟情况下获得client梯度数据。|
|Hetero LinR|纵向线性回归[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/linear_regression/#heterogeneous-linr)|Party A为guest且拥有label，Party B为Host，需要第三方arbiter| 梯度聚合可采用paillier同态加密、秘密分享协议。若采用paillier同态加密，为避免arbiter获取梯度中间数据，需增加附加掩码；若采用秘密分享协议，要求至少具有两个秘密分享server，且server之间不能共谋。|
|Hetero Federated Poisson Regression |纵向泊松回归 [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/poisson_regression/)|Party A为guest且拥有label，Party B为Host，需要第三方arbiter|梯度聚合可采用paillier同态加密、秘密分享协议。若采用paillier同态加密，为避免arbiter获取梯度中间数据，需增加附加掩码；若采用秘密分享协议，要求至少具有两个秘密分享server，且server之间不能共谋。|
|Homo Neural Network|横向神经网络[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/homo_nn/)|参与方身份对等，需要第三方arbiter|安全聚合采用秘密分享协议，并且t>n/2（t是指恢复密码的最小份数，n为client总数？）；为防止server虚构client数量，推荐使用PKI为client提供注册信息；采用double-mask防止server在处理掉线client或者网络延迟情况下获得client梯度数据。|
|Hetero Secure Boosting|异构安全梯度提升决策树 [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#hetero-secureboost )|Active Party 为dominating server，且拥有label y|由Active Party生成Paillier Encryption密钥保护gi和hi|
|Hetero NN| [link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_nn/)|Party A提供数据矩阵，Party B提供数据矩阵和label y，且为dominating server|由Party B生成Paillier Encryption密钥保护activations，同时Party A、B对activations添加噪声防止推理。 |
|Homo Secure Boost|[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/ensemble/#homo-secureboost)|参与方身份对等，需要第三方arbiter|Clients之间协商总和可抵消归零的随机数，附加到g和h中，使server无法知悉每个client的训练中间数据。|
|Hetero FTL|[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_ftl/)|Host方拥有label，不需要第三方arbiter|Host和Client同时生成Paillier Encryption密钥和掩码保护训练中间数据。|
|Hetero KMeans|[link](https://fate.readthedocs.io/en/latest/zh/federatedml_component/hetero_kmeans/)|参与方对等，需要第三方arbiter|Clients之间协商总和可抵消归零的随机数，附加到训练中间数据中，使server无法知悉每个client的训练中间数据。|


## 五. 合规性建议
FATE本身是一个开源的联邦机器学习平台，各参与方通过安全协议等交换中间数据，确保自身的数据不出域，最终达到共同计算机器学习模型的目的。当FATE架构用于生产实践中时，一些与信息安全、网络安全、数据安全相关的法律法规有助于使用者从安全、合规角度更有效的使用FATE。以下列出了使用FATE过程中可能涉及的国内外法律法规，供参考。

### 国内相关法律法规
 - 中华人民共和国数据安全法
 - 中华人民共和国个人信息保护法
 - 中华人民共和国网络安全法
 - PIA GB∕T 39335-2020 信息安全技术 个人信息安全影响评估指南

### 国际法律法规

 - GDPR-General Data Protection Regulation
 - CCPA-California Consumer Privacy Act Regulation
 - HIPAA-Health Insurance Portability and Accountability Act of 1996
 - DPIA-Data Protection Impact Assessment







