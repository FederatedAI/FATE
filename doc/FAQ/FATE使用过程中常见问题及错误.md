1. **运行任务的日志出现loss:inf的报错**
这个一般是由于输入数据(或者部分数据)没有经过归一化导致的，因为这里求梯度和loss是在0点附近做泰勒展开，如果数据没有做归一化，就容易发散了

2. **log里面出现“Count of data_instance is 0”**
说明该角色方没有导入数据，或者数据表配置有问题

3. **data_io组件一直停在 DenseFeatureReader.gen_data_instance() 是什么原因呢？**
查看卡住一方的roll下面是否报错，一般是dataio的配置有问题，导致数据转换在执行过程中出错

4. **请问数据上传之后是存到了默认的数据库里了吗？怎么查看呢**
默认的数据库应该只是存下了数据的元信息，数据存放的文件夹是fate/data-dir/lmdb/{job_id}.
如果要查看数据可以用 python fate_flow_client.py -f download python fate_flow_client.py -f download -c download.json （参考https://github.com/FederatedAI/FATE/tree/master/examples/federatedml-1.0-examples）

5. **如果guest方只有Y，没有任何X，guest方的所有特征处理及模型训练是要设置need_run为false吗？**
特征工程的组件，默认need_run为true即可，训练组件，如lr和secureboost是不能设置为false的，可以在guest那边，手动加一列X，X的取值很小，比如0.00001这样，但是这样做对guest可能存在泄露y的风险，不太建议这样做。

6. **预测阶段用的好像是带label训练数据？如果想对仅带feature的数据做预测如何修改呢？**
现在的预测所用模型和配置是训练的时候生成的，像dataio这个数据转换组件，会解析label\features，预测的时候执行的是同样的流程，所以如果仅带features数据的话，需要自己在数据里面加一列伪label。

7. **fate 数据以什么编码格式存储到 lmdb 中的？**
目前是pickle后存储进去的

8. **请问FATE为什么会需要这么大的磁盘空间？**
即将发布FATE-1.1的版本会对这个进行改善，在每个epoch之后会做中间数据的清理。

9. **如何查看train和predict过程中各方交互信息？**
点击FATE boardd的job界面右上角的dashboard，然后看到下面LOG，DEBUG级别的日志里面cluster.py-[linexxxx] [REMOTE] / [GET](standalone版本的话的话，cluster.py=>standalone.py) ，分别表示发送和接受信息。不过board上的DEBUG日志是job级别的，如果要具体到某个组件，如hetero_lr_0，需要到fate_flow_server的部署机器上的logs/$jobid/$role/$partyid下看DEBUG.log

10. **请问有什么方式能够删除一些测试任务占用的存储空间？**
standalone在 data目录下，可以删除IN_MEMORY。 cluster的话data-dir目录的IN_MEMORY。
