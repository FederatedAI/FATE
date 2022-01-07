# 使用教程

这里我们提供一些关于跑FATE任务的教程:

我们推荐使用 `Pipeline` 来提交任务, 这里提供一些 `Jupyter Notebook` 来交互式演示如何使用`Pipeline`来提交任务:
    
- [用 `Pipeline` 上传数据](pipeline/pipeline_tutorial_upload.ipynb)
- [用 `Pipeline` 进行 `Hetero SecureBoost` 训练和预测](pipeline/pipeline_tutorial_hetero_sbt.ipynb)
- [用 `Pipeline` 构建横向神经网络模型](pipeline/pipeline_tutorial_homo_nn.ipynb)
- [用 `Pipeline` 进行带 `Match ID` 的 `Hetero SecureBoost` 训练和预测](pipeline/pipeline_tutorial_match_id.ipynb)

不使用 `Pipeline` 来提交任务也是支持的，用户需要配置一些 `json` 格式的任务配置文件:

- [教程](dsl_conf/dsl_conf_tutorial.md)
- [上传数据](dsl_conf/upload_data_guide.md)
- [任务配置](dsl_conf/dsl_conf_v2_setting_guide.md)

用 `FATE Serving` 发布模型:

- [发布模型](model_publish_with_serving_guide.md)

用 `FATE-Test` 跑多个任务:
    
- [FATE-Test 教程](fate_test_tutorial.md)
