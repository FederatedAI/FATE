# 连接FATE flow
# !pipeline init --ip 127.0.0.1 --port 9380
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data
def train_pipeline():
    # 创建pipeline实例
    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=10000)
    # load data
    reader_0 = Reader(name="reader_0")
    # set guest parameter
    reader_0.get_party_instance(role='guest', party_id=9999).component_param(
        table={"name": "breast_hetero_guest", "namespace": "experiment"})
    # set host parameter
    reader_0.get_party_instance(role='host', party_id=10000).component_param(
        table={"name": "breast_hetero_host", "namespace": "experiment"})
    # 添加DataTransform组件以将原始数据解析到数据实例中
    data_transform_0 = DataTransform(name="data_transform_0")
    # set guest parameter
    data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
        with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
        with_label=False)
    # 添加求交组件以执行纵向场景的PSI
    intersect_0 = Intersection(name="intersect_0")
    # 定义HeteroSecureBoost组件。将为所有相关方设置以下参数。
    hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                             num_trees=5,
                                             bin_num=16,
                                             task_type="classification",
                                             objective_param={"objective": "cross_entropy"},
                                             encrypt_param={"method": "paillier"},
                                             tree_param={"max_depth": 3})
    # 模型评估组件
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    # 将组件添加到管道，按执行顺序
    # -data_transform_0 用 reader_0的输出数据
    # -intersect_0 用 data_transform_0的输出数据
    # -heter_securebost_0用intersect_0的输出数据
    # -评估0用纵向securebost0对训练数据的预测结果
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))
    pipeline.compile()
    # submit pipeline
    pipeline.fit()
    # 一旦完成了训练，就可以将训练后的模型用于预测。（可选）保存经过训练的管道以备将来使用。
    # 模型保存
    pipeline.dump("hetero_secureboost_pipeline_saved.pkl")
def predict_pipeline():
    # 模型预测
    # 首先，从训练管道部署所需组件
    pipeline = PipeLine.load_model_from_file('hetero_secureboost_pipeline_saved.pkl')
    pipeline.deploy_component([pipeline.data_transform_0, pipeline.intersect_0, pipeline.hetero_secureboost_0])
    # 定义用于读取预测数据的新reader组件
    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role="guest", party_id=9999).component_param(table={"name": "breast_hetero_guest", "namespace": "experiment"})
    reader_1.get_party_instance(role="host", party_id=10000).component_param(table={"name": "breast_hetero_host", "namespace": "experiment"})
    # （可选）定义新的评估组件。
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    # 按执行顺序添加组件到预测管道
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_1)\
                    .add_component(pipeline,
                                   data=Data(predict_input={pipeline.data_transform_0.input.data: reader_1.output.data}))\
                    .add_component(evaluation_0, data=Data(data=pipeline.hetero_secureboost_0.output.data))
    # 执行预测
    predict_pipeline.predict()

if __name__=='__main__':
    # train_pipeline()
    predict_pipeline()







