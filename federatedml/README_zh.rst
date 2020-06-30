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

.. figure:: ../doc/images/federatedml_structure.png
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

   * - `DataIO`_
     - DataIO
     - 该组件通常是建模任务的第一个组件。 它将用户上传的数据转换为Instance对象，该对象可用于以下组件。
     - DTable，值为原始数据
     - 转换后的数据表，值为在 :download:`federatedml/feature/instance.py <feature/instance.py>` 中定义的Data Instance的实例
     - 
     -

   * - `Intersect`_
     - Intersection
     - 计算两方的相交数据集，而不会泄漏任何差异数据集的信息。主要用于纵向任务。
     - DTable
     - 两方DTable中相交的部分
     - 
     -

   * - `Federated Sampling`_
     - FederatedSample
     - 对数据进行联邦采样，使得数据分布在各方之间变得平衡。这一模块同时支持单机和集群版本。
     - DTable
     - 采样后的数据，同时支持随机采样和分层采样
     - 
     -

   * - `Feature Scale`_
     - FeatureScale
     - 特征归一化和标准化。
     - DTable，其值为instance
     - 转换后的DTable
     - 变换系数，例如最小值/最大值，平均值/标准差
     -

   * - `Hetero Feature Binning`_
     - Hetero Feature Binning
     - 使用分箱的输入数据，计算每个列的iv和woe，并根据合并后的信息转换数据。
     - DTable，在guest中有标签y，在host中没有标签y
     - 转换后的DTable
     - 
     - 每列的iv/woe，分裂点，事件计数，非事件计数等
   
   * - `OneHot Encoder`_
     - OneHotEncoder
     - 将一列转换为One-Hot格式。
     - 数据输入：DTable
     - 转换了带有新列名的DTable
     - 
     - 原始列名和特征值到新列名的映射
    
   * - `Hetero Feature Selection`_
     - HeteroFeatureSelection
     - 提供多种类型的filter。每个filter都可以根据用户配置选择列。
     - DTable
     - 转换的DTable具有新的header和已过滤的数据实例
     - 模型输入如果使用iv filters，则需要hetero_binning模型
     - 每列是否留下
   
   * - `Union`_
     - Union
     - 将多个数据表合并成一个。
     - DTables
     - 多个Dtables合并后的Dtable
     - 
     -

   * - `Hetero-LR`_
     - HeteroLR
     - 通过多方构建纵向逻辑回归模块。
     - DTable
     - 
     -
     - Logistic回归模型
   
   * - `Local Baseline`_
     - LocalBaseline
     - 使用本地数据运行sklearn Logistic回归模型。
     - DTable
     - 
     -
     - Logistic回归
   
   * - `Hetero-LinR`_
     - HeteroLinR
     - 通过多方建立纵向线性回归模块。
     - DTable
     - 
     -
     - 线性回归模型
   
   * - `Hetero-Poisson`_
     - HeteroPoisson
     - 通过多方构建纵向泊松回归模块。
     - DTable
     - 
     -
     - 泊松回归模型
   
   * - `Homo-LR`_
     - HomoLR
     - 通过多方构建横向逻辑回归模块。
     - DTable
     - 
     -
     - Logistic回归模型
   
   * - `Homo-NN`_
     - HomoNN
     - 通过多方构建横向神经网络模块。
     - DTable
     - 
     -
     - 神经网络模型
    
   * - `Hetero Secure Boosting`_
     - HeteroSecureBoost
     - 通过多方构建纵向Secure Boost模块。
     - DTable，其值为instance
     - 
     -
     - SecureBoost模型，由模型本身和模型参数组成
    
   * - `Evaluation`_
     - Evaluation
     - 为用户输出模型评估指标。
     - 
     -
     -
     -

   * - `Hetero Pearson`_
     - HeteroPearson
     - 计算来自不同方的特征的Pearson相关系数。
     - DTable
     - 
     -
     -
    
   * - `Hetero-NN`_
     - HeteroNN
     - 构建纵向神经网络模块。
     - DTable
     - 
     -
     - 纵向神经网络模型
    
   * - `Homo Secure Boosting`_
     - HomoSecureBoost
     - 通过多方构建横向Secure Boost模块
     - DTable, 其值为instance
     - 
     - 
     - SecureBoost模型，由模型本身和模型参数组成



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


.. toctree::
   :maxdepth: 2

   util/README
   statistic/intersect/README
   feature/README
   statistic/union/README
   linear_model/logistic_regression/README
   local_baseline/README
   linear_model/linear_regression/README
   linear_model/poisson_regression/README
   nn/homo_nn/README
   tree/README
   evaluation/README
   statistic/correlation/README
   nn/hetero_nn/README
   model_selection/stepwise/README




安全协议
---------


.. toctree::
   :maxdepth: 2

   secureprotol/README


