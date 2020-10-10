联邦机器学习
============
[`ENG`_]

.. _ENG: README.rst

Federatedml模块包括许多常见机器学习算法联邦化实现。所有模块均采用去耦的模块化方法开发，以增强模块的可扩展性。具体来说，我们提供：

1. 联邦统计: 包括隐私交集计算，并集计算，皮尔逊系数等

2. 联邦特征工程：包括联邦采样，联邦特征分箱，联邦特征选择等。

3. 联邦机器学习算法：包括横向和纵向的联邦LR, GBDT， DNN，迁移学习等

4. 模型评估：提供对二分类，多分类，回归评估，联邦和单边对比评估

5. 安全协议：提供了多种安全协议，以进行更安全的多方交互计算。

.. figure:: ../../doc/images/federatedml_structure.png
   :align: center
   :width: 800
   :alt: 联邦学习结构

   Figure 1： Federated Machine Learning Framework


算法清单
--------

.. list-table:: 算法清单
   :widths: 10 10 40 10 10 10 10
   :header-rows: 1

   * - 算法
     - 模块名
     - 描述
     - 数据输入
     - 数据输出
     - 模型输入
     - 模型输出

   * - Reader
     - Reader
     - 当输入数据的存储引擎当前计算引擎不支持时，会自动转存到FATE集群适配计算引擎的组件输出存储引擎；当输入数据的存储格式非FATE支持存储格式时，会自动转换格式，并存储到FATE集群的组件输出存储引擎
     - 用户原始存储数据
     - 转换后原始数据
     -
     -

   * - `DataIO`_
     - DataIO
     - 该组件将原始数据转换为Instance对象。
     - Table，值为原始数据
     - 转换后的数据表，值为在 : `federatedml/feature/instance.py` 中定义的Data Instance的实例
     -
     - DataIO模型

   * - `Intersect`_
     - Intersection
     - 计算两方的相交数据集，而不会泄漏任何差异数据集的信息。主要用于纵向任务。
     - Table
     - 两方Table中相交的部分
     -
     - Intersect模型

   * - `Federated Sampling`_
     - FederatedSample
     - 对数据进行联邦采样，使得数据分布在各方之间变得平衡。这一模块同时支持单机和集群版本。
     - Table
     - 采样后的数据，同时支持随机采样和分层采样
     -
     -

   * - `Feature Scale`_
     - FeatureScale
     - 特征归一化和标准化。
     - Table，其值为instance
     - 转换后的Table
     - 变换系数，例如最小值/最大值，平均值/标准差
     -

   * - `Hetero Feature Binning`_
     - Hetero Feature Binning
     - 使用分箱的输入数据，计算每个列的iv和woe，并根据合并后的信息转换数据。
     - Table，在guest中有标签y，在host中没有标签y
     - 转换后的Table
     -
     - 每列的iv/woe，分裂点，事件计数，非事件计数等
   
   * - `OneHot Encoder`_
     - OneHotEncoder
     - 将一列转换为One-Hot格式。
     - Table, 值为Instance
     - 转换了带有新列名的Table
     -
     - 原始列名和特征值到新列名的映射
    
   * - `Hetero Feature Selection`_
     - HeteroFeatureSelection
     - 提供多种类型的filter。每个filter都可以根据用户配置选择列。
     - Table, 值为Instance
     - 转换的Table具有新的header和已过滤的数据实例
     - 模型输入如果使用iv filters，则需要hetero_binning模型
     - 每列是否留下
   
   * - `Union`_
     - Union
     - 将多个数据表合并成一个。
     - Tables
     - 多个Tables合并后的Table
     -
     -

   * - `Hetero-LR`_
     - HeteroLR
     - 通过多方构建纵向逻辑回归模块。
     - Table, 值为Instance
     - 
     -
     - Logistic回归模型，由模型本身和模型参数组成
   
   * - `Local Baseline`_
     - LocalBaseline
     - 使用本地数据运行sklearn Logistic回归模型。
     - Table, 值为Instance
     - 
     -
     -
   
   * - `Hetero-LinR`_
     - HeteroLinR
     - 通过多方建立纵向线性回归模块。
     - Table, 值为Instance
     - 
     -
     - 线性回归模型，由模型本身和模型参数组成
   
   * - `Hetero-Poisson`_
     - HeteroPoisson
     - 通过多方构建纵向泊松回归模块。
     - Table, 值为Instance
     - 
     -
     - 泊松回归模型，由模型本身和模型参数组成
   
   * - `Homo-LR`_
     - HomoLR
     - 通过多方构建横向逻辑回归模块。
     - Table, 值为Instance
     -
     -
     - Logistic回归模型，由模型本身和模型参数组成
   
   * - `Homo-NN`_
     - HomoNN
     - 通过多方构建横向神经网络模块。
     - Table, 值为Instance
     - 
     -
     - 神经网络模型，由模型本身和模型参数组成
    
   * - `Hetero Secure Boosting`_
     - HeteroSecureBoost
     - 通过多方构建纵向Secure Boost模块。
     - Table，值为Instance
     - 
     -
     - SecureBoost模型，由模型本身和模型参数组成
    
   * - `Evaluation`_
     - Evaluation
     - 为用户输出模型评估指标。
     - Table(s), 值为Instance
     -
     -
     -

   * - `Hetero Pearson`_
     - HeteroPearson
     - 计算来自不同方的特征的Pearson相关系数。
     - Table, 值为Instance
     - 
     -
     -
    
   * - `Hetero-NN`_
     - HeteroNN
     - 构建纵向神经网络模块。
     - Table, 值为Instance
     - 
     -
     - 纵向神经网络模型
    
   * - `Homo Secure Boosting`_
     - HomoSecureBoost
     - 通过多方构建横向Secure Boost模块
     - Table, 值为Instance
     - 
     - 
     - SecureBoost模型，由模型本身和模型参数组成

   * - `Homo OneHot Encoder`_
     - 横向 OneHotEncoder
     - 将一列转换为One-Hot格式。
     - Table, 值为Instance
     - 转换了带有新列名的Table
     -
     - 原始列名和特征值到新列名的映射

   * - `Data Split`_
     - 数据切分
     - 将输入数据集按用户自定义比例或样本量切分为3份子数据集
     - Table, 值为Instance
     - 3 Tables
     -
     -

   * - `Column Expand`_
     -
     - 对原始Table添加任意列数的任意数值
     - Table, 值为原始数据
     - 转换后带有新数列与列名的Table
     -
     - Column Expand模型

   * - `Secure Information Retrieval`_
     -
     - 通过不经意传输协议安全取回所需数值
     - Table, 值为Instance
     - Table, 值为取回数值
     -
     -

   * - `Hetero KMeans`_
     - 纵向 K均值算法
     - 构建K均值模块
     - Table, 值为Instance
     - Table, 值为Instance; Arbiter方输出2个Table
     -
     - Hetero KMeans模型

   * - `Scorecard`_
     - 评分卡
     - 转换二分类预测分数至信用分
     - Table, 值为二分类预测结果
     - Table, 值为转化后信用分结果
     -
     -

.. _DataIO: util/README.rst
.. _Intersect: statistic/intersect/README.rst
.. _Federated Sampling: feature/README.rst
.. _Feature Scale: feature/README.rst
.. _Hetero Feature Binning: feature/README.rst
.. _OneHot Encoder: feature/README.rst
.. _Hetero Feature Selection: feature/README.rst
.. _Union: statistic/union/README.rst
.. _Hetero-LR: linear_model/logistic_regression/README.rst
.. _Local Baseline: local_baseline/README.rst
.. _Hetero-LinR: linear_model/linear_regression/README.rst
.. _Hetero-Poisson: linear_model/poisson_regression/README.rst
.. _Homo-LR: linear_model/logistic_regression/README.rst
.. _Homo-NN: nn/homo_nn/README.rst
.. _Hetero Secure Boosting: tree/README.rst
.. _Evaluation: evaluation/README.rst
.. _Hetero Pearson: statistic/correlation/README.rst
.. _Hetero-NN: nn/hetero_nn/README.rst
.. _Homo Secure Boosting: tree/README.rst
.. _Data Split: model_selection/data_split/README.rst
.. _Homo OneHot Encoder: feature/README.rst
.. _Column Expand: feature/README.rst
.. _Secure Information Retrieval: secure_information_retrieval
.. _Hetero KMeans: unsupervised_learning/kmeans/README.rst
.. _Scorecard: statistic/scorecard/README.rst


安全协议
---------


* `Encrypt`_

   - `Paillier encryption`_
   - `Affine Homomorphic Encryption`_
   - `IterativeAffine Homomorphic Encryption`_
   - `RSA encryption`_
   - `Fake encryption`_

* `Encode`_

* `Diffne Hellman Key Exchange`_

* `SecretShare MPC Protocol(SPDZ)`_

* `Oblivious Transfer`_


.. _Encrypt: secureprotol/README.rst#encrypt
.. _Paillier encryption: secureprotol/README.rst#paillier-encryption
.. _Affine Homomorphic Encryption: secureprotol/README.rst#affine-homomorphic-encryption
.. _IterativeAffine Homomorphic Encryption: secureprotol/README.rst#iterativeaffine-homomorphic-encryption
.. _RSA encryption: secureprotol/README.rst#rst-encryption
.. _Fake encryption: secureprotol/README.rst#fake-encryption
.. _Encode: secureprotol/README.rst#encode
.. _Diffne Hellman Key Exchange: secureprotol/README.rst#diffne-hellman-key-exchange
.. _SecretShare MPC Protocol(SPDZ): secureprotol/README.rst#secretshare-mpc-protocol-spdz
.. _Oblivious Transfer: secureprotol/README.rst#oblivious-transfer



Params
-------

.. automodule:: federatedml.param
   :autosummary:
   :members:
