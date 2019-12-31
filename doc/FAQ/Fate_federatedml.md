1. **每个算法组件的参数列表应该去哪里查看？**
我们在github上有参数列表文件：
https://github.com/FederatedAI/FATE/tree/master/federatedml/param
同时在example下各个算法中，也有例子的配置文件可供大家参考。
后续会有专门的文档页面。

2. **federatedml相关的文档都有哪些？**
我们在github的federatedml文件夹下有各个算法的列表，对应算法有链接连到具体算法的文档中。
具体可以参考：
https://github.com/FederatedAI/FATE/tree/master/federatedml

3. **homo_lr或者hetero_lr的原理？**
请查看github上的算法介绍：
https://github.com/FederatedAI/FATE/blob/master/federatedml/logistic_regression/README.md
我们的官网上也有详细介绍。

4. **Guest,Host, Arbiter分别是什么?**
Guest表示数据应用方，Host是数据提供方，在纵向算法中，Guest往往是有标签y的一方。arbiter是用来辅助多方完成联合建模的，主要的作用是用来聚合梯度或者模型,比如纵向lr里面,各方将自己一半的梯度发送给arbiter，然后arbiter再联合优化等等,arbiter还参与以及分发公私钥，进行加解密服务等等

5. **为什么只能是Guest发起任务?**
因为Guest是数据应用方，在纵向模型的建模中往往是有y的一方。通常整个建模任务通常是服务于y的业务逻辑的，因此也只有guest需要应用这一模型。同时，如果允许Host方任意发起建模流程，有可能会有Guest方的数据泄露风险。

6. **如何对建好的模型做离线预测?**
在训练好模型以后，记录下model version和model id(可以在提交任务的runtime_conf下面找到，或者根据fate flow接口按照jobid查询到)，然后在example下，有一个样例的predict配置，将两项内容填进去，同时配好多方信息，再启动任务即可。
具体流程可参考文档:
https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples#step3-start-your-predict-task

7. **现在哪些算法支持多个host建模了呢**
目前交集(Intersect)，横向和纵向的逻辑回归(Homo/Hetero-LR)，纵向的线性回归(Hetero-LinR)、纵向SecureBoost,联邦特征分箱和联邦特征选择均已支持多个数据提供方(Host)参与建模。

8. **多host的任务配置如何配**
各支持多host的模块均在https://github.com/FederatedAI/FATE/tree/master/examples 里提供多host配置样例。可以参考样例调整配置。

9. **如何使用Spark模式**
	1. 首先需要部署yarn环境，并把SPARK_HOME加入fate_flow启动脚本
	2. 在conf的job_parameters中加入”backend”: 1

10. **支持哪些Spark提交模式**
目前仅支持Spark-Client模式

11. **最多可以多少方参与？ 性能如何？是否可以一方发起所有操作？**
答: 目前横向可以支持多方，单机版本至少10个并行跑过没问题（分布式版本，目前没用那么多机器测试）。多方的性能数据暂时还没统计。目前就是FATE就是通过一方来发起所有任务，其他方不需要任何手动操作。

12. **如何对参与联邦的各方中的数据质量进行评价，以避免脏数据对联邦模型的影响?**
可以通过对加入方的数据对模型提升效果来评判。目前参与横向联邦的数据需要在各方中全部对齐，后续如何解决仅在本方中独有的数据参与联邦学习。这个问题可以通过联邦迁移学习来解决

13. **目前1.1版本的 纵向lr好像有点慢，一次迭代通信需要3次，再加上同态加密时间开销好像很大。请问目前有比较快的策略嘛？**
相对1.0, 1.1版本已经把llr通信量降了很多，此外lr参数加密模式那里可以采用用fast模式，会快很多，内部测差不多提升一倍

