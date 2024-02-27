# 联邦机器学习

Federatedml模块包括许多常见机器学习算法联邦化实现。所有模块均采用去耦的模块化方法开发，以增强模块的可扩展性。具体来说，我们提供：

1. 联邦统计: 包括隐私交集计算，并集计算
2. 联邦特征工程：包括联邦采样，联邦特征分箱，联邦特征选择等。
3. 联邦机器学习算法：包括横向和纵向的联邦LR, GBDT， DNN等
4. 模型评估：提供对二分类，多分类，回归评估
5. 安全协议：提供了多种安全协议，以进行更安全的多方交互计算。

## 算法清单

如需不通过FATE-Client直接调用算法模块，请查看此[教程](../ml/run_launchers.md).

| 算法                                               | 模块名                    | 描述                                         | 数据输入                                          | 数据输出                                                      | 模型输入                                   | 模型输出               |
|--------------------------------------------------|------------------------|--------------------------------------------|-----------------------------------------------|-----------------------------------------------------------|----------------------------------------|--------------------|
| [Reader](reader.md)                              |                        | 传递用户指定输入数据表给下游组件                           |                                               | output_data                                               |                                        |                    |
| [PSI](psi.md)                                    | PSI                    | 计算两方的相交数据集，而不会泄漏任何差异数据集的信息。主要用于纵向任务        | input_data                                    | output_data                                               |                                        |                    |
| [Sampling](sample.md)                            | Sample                 | 对数据进行联邦采样，使得数据分布在各方之间变得平衡。这一模块同时支持本地和联邦场景。 | input_data                                    | output_data                                               |                                        |                    |
| [Data Split](data_split.md)                      | DataSplit              | 将数据集切分成训练、验证、测试集。                          | input_data                                    | train_output_data, validate_output_data, test_output_data |                                        |                    |
| [Feature Scale](feature_scale.md)                | FeatureScale           | 特征归一化和标准化。                                 | train_data, test_data                         | train_output_data, test_output_data                       | input_model                            | output_model       |
| [Data Statistics](statistics.md)                 | Statistics             | 计算各类统计指标。                                  | input_data                                    |                                                           |                                        | output_model       |
| [Hetero Feature Binning](feature_binning.md)     | HeteroFeatureBinning   | 使用分箱的输入数据，计算每个列的iv和woe，并根据合并后的信息转换数据。      | train_data, test_data                         | train_output_data, test_output_data                       | input_model                            | output_model       |
| [Hetero Feature Selection](feature_selection.md) | HeteroFeatureSelection | 提供多种类型的filter。每个filter都可以根据用户配置选择列。        | train_data, test_data                         | train_output_data, test_output_data                       | input_models, input_model              | output_model       |
| [Coordinated-LR](logistic_regression.md)         | CoordinatedLR          | 通过多方构建纵向逻辑回归模块。                            | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
| [Coordinated-LinR](linear_regression.md)         | CoordinatedLinR        | 通过多方建立纵向线性回归模块                             | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
| [Homo-LR](logistic_regression.md)                | HomoLR                 | 通过多方构建横向逻辑回归模块。                            | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
| [Homo-NN](homo_nn.md)                            | HomoNN                 | 通过多方构建横向神经网络模块。                            | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
| [Hetero-NN](hetero_nn.md)                        | HeteroNN               | 通过多方构建纵向联邦神经网络模型。                          | train_data, validate_data, test_data          | train_data_output, predict_data_output                    | train_model_input, predict_model_input | train_model_output |
| [Hetero Secure Boosting](hetero_secureboost.md)  | HeteroSecureBoost      | 通过多方构建纵向联邦梯度提升树模型。                         | train_data, test_data, cv_data                | train_data_output, test_data_output, cv_output_datas      | train_model_input, predict_model_input | train_model_output |
| [Evaluation](evaluation.md)                      | Evaluation             | 评估二分类、多分类、回归等指标。                           | input_data                                    |                                                           |                                        |                    |
| [Union](union.md)                                | Union                  | 将多个数据表合并成一个。                               | input_data                                    | output_data                                               |                                        |                    |
| [SSHE-LR](logistic_regression.md)                | SSHELR                 | 通过两方构建纵向逻辑回归模块。                            | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
| [SSHE-LinR](linear_regression.md)                | SSHELinR               | 通过两方构建纵向线性回归模块。                            | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas      | input_model, warm_start_model          | output_model       |
