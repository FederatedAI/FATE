### Param

#### 1. Class DataIOParam

```
__init__(delimitor=“,”, data_type=“float64”, missing_fill=False,
        default_value=0, with_label=False, label_idx=0,
        label_type=None, output_format=‘dense’)
```
类描述: data_io类参数,给data_io.DenseFeatureReader/SparseFeatureReader所用

Params:
1. delemitor: str，分隔符
2. data_type: str，列数据类型
3. missing_fill: bool，是否需要填充缺失值
4. default_value: int/float，缺失值填充字段
5. with_label: bool，数据是否包含标签列
6. label_idx: int，若含标签列,列下标
7. label_type: str，标签类型
8. output_format: str，将输出转化成稠密、稀疏存储

#### 2. Class LogisticParam
```
__init__(penalty=”L2”, fit_intercept=True,
        eps=1e-5, alpha=1.0, optimizer=”sgd”, batch_size=-1,
        learning_rate=0.01, max_iter=100, converge_func=’difference’,
        communication_param=CommunicationParam(), re_encrypt_batches=2,
        model_db=’lr_model’, model_table=’lr_table’, party_weight=1.0,
        init_params=InitParam(), predict_threshold=0.5)
```
1)	penalty: str, “L1” or “L2” or None, default:”L2”.
    The norm used in penalization term.
2)	init_params: InitParam object.
3)	eps: float, default: 1e-5
    Tolerance for judging convergence.
4)	alpha: float, default: 1.0
    Regularization strength. Must be a positive float.
5)	optimizer: str, default: "sgd"
    Optimization method. Available terms: {"sgd", "rmsprop", "

6)	batch_size: int, default:-1. bat
7)	learning_rate: float, default: 0.01 , learning rate
8)	max_iter: int, default: 100. Max number of iteration
9)	converge_func: str, default: “difference”. Must be in “diff” or “abs”. Both method use eps as threshold
a)	diff： Use difference of loss between two iterations to judge whether converge.
b)	abs: Use the absolute value of loss to judge whether converge.
10)	communication_param: CommunicationParam object. Use to store party_ids.
11)	use_encrypt: bool, default: True. Specify whether host use encryption or not.
12)	re_encrypt_batches: int, default: 2. Specify how many batches to do an re-encrypt operation. Re-encryption is needed since multiple add and multiply operations would cause overflow in Pairllier encryption.
13)	party_weight: floats, default: 1.0. Specify the model weight of itself. This weight means the importance of this party. The higher of this weight, means higher ratio will this party get when aggregating models.
14)	model_db: str, default: “model_db”. The db name to save and load this model.
15)	model_table: str, default: “model_table”. The table name of this model.
16)	predict_threshold: float, default: 0.5. The threshold for judging class 0 or 1.

