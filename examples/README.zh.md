# Example 使用指南 
[English](README.md)


本章节介绍examples目录，它提供了pipeline样例、dsl的配置、 以及正确性对比验证回归的样例、常规建模模版等。

为方便用户对FATE进行自动化测试，FATE提供自动化测试工具[FATE-Test](../doc/api/fate_test.md)。

为方便使用Pipeline或者DSL提交建模任务，我们建议用户安装使用[FATE-Client](../doc/api/fate_client/pipeline.md)。

快速上手Pipeline/DSL，可以参考[快速使用教程](../doc/tutorial/README.md)

下面将具体介绍主要的样例模块。

## FATE-Pipeline

为了提升联邦建模的易用性，FATE-v1.5 开始引入Pipeline模块， 用户可通过python编程快速搭建联邦学习建模流程，具体教程可参考
[FATE-Pipeline](../doc/api/fate_client/pipeline.md)。
我们对于每个算法模块也提供了大量的Pipeline搭建联邦学习模型的样例，具体可参考[pipeline](./pipeline)。

## DSL

DSL是FATE提供的根据配置文件来构建联邦建模任务的方式，具体教程可参考
[DSL配置指引](../doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide.md)。
在FATE-v1.5版本，我们对DSL进行了全新升级.
    主要升级点包括下面几点：
1.  支持按需生成预测DSL，用户可通过FATE-Flow的cli按需生成预测的DSL配置，需要注意的是新版DSL不支持自动预测DSL的生成，用户必须先通过FATE-Flow的cli生成预测DSL，然后再进行预测
2.  支持预测阶段新组件接入，如在预测阶段接入evaluation组件等。
3.  统一算法参数配置风格，role\_parameter和algorithm\_parameter规范统一

关于最新的DSL各算法组件样例可参考 [dsl/v2](./dsl/v2)，自从1.7版本开始DSL/v1被删除，不再继续支持，但是会提供DSL/v1构建的模型转换为DSL/v2的工具，方便模型迁移到DSL/v2。


## Benchmark Quality

从FATE-v1.5开始，FATE将提供中心化训练和FATE联邦建模效果的正确性对比工具，用于算法的正确性对比。
我们优先提供了建模中最常用的算法的正确性对比脚本，包括以下模型类型： 
- 纵向:
    - LogisticRegression([benchmark\_quality/hetero\_lr](./benchmark_quality/hetero_lr))
    - LinearRegression([benchmark\_quality/hetero\_linear_regression](./benchmark_quality/hetero_linear_regression))
    - SecureBoost([benchmark\_quality/hetero\_sbt](./benchmark_quality/hetero_sbt))
    - FastSecureBoost([benchmark\_quality/hetero\_fast\_sbt](./benchmark_quality/hetero_fast_sbt)),
    - NN([benchmark\_quality/hetero\_nn](./benchmark_quality/hetero_nn))
- 横向:
    - LogisticRegression([benchmark\_quality/homo\_lr](./benchmark_quality/homo_lr))
    - SecureBoost([benchmark\_quality/homo\_sbt](./benchmark_quality/homo_sbt))
    - NN([benchmark\_quality/homo\_nn](./benchmark_quality/homo_nn)

执行方法可参考[benchmark\_quality使用文档](../doc/api/fate_test.md#benchmark-quality)


## Benchmark Performance

FATE-Test 同时支持FATE联邦学习模型效率benchmark测试。
我们提供了以下模型的benchmark测试集：

  - Hetero Logistic Regression([benchmark\_performance/hetero\_lr](./benchmark_performance/hetero_lr)),
  - Hetero SecureBoost([benchmark\_performance/hetero\_sbt](./benchmark_performance/hetero_sbt)),
  - Hetero SSHE LR([benchmark\_performance/hetero\_fast\_sbt](./benchmark_performance/hetero_sshe_lr)),
  - Hetero Intersect:
    - [benchmark\_performance/intersect_single](./benchmark_performance/intersect_single)
    - [benchmark\_performance/intersect_multi](./benchmark_performance/intersect_multi)
  
执行方法可参考[benchmark\_performance使用文档](../doc/api/fate_test.md#benchmark-performance).


## Upload Default Data

用户可使用FATE-Test [data](../doc/api/fate_test.md#data)上传数据。

## Toy Example

为了方便用户快速测试站点间网络连通性，FATE提供了简洁的toy任务，使用方法类似：
```
flow test toy -gid 9999 -hid 10000
```

## Min-test

为了方便用户体验建模流程，检测部署完成情况，FATE提供了最小化测试脚本，方便用户一键体验。该脚本将启动纵向逻辑回归和纵向secure\_boost算法。用户只需一行启动命令，
配置若干简单参数，即可完成全流程建模。具体使用方法，请参考[min\_test\_task](./min_test_task/README.rst)
