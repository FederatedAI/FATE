from pipeline.components.fate import Reader
from pipeline.components.fate import Intersection
from pipeline.pipeline import StandalonePipeline
from pipeline.components.fate import FeatureScale


pipeline = StandalonePipeline().set_scheduler_party_id(party_id=10001).set_roles(
        guest=9999, host=[10000, 10001], arbiter=10001)
reader_0 = Reader(name="reader_0")
reader_0.guest.component_param(path="examples/data/breast_hetero_guest.csv",
                               format="csv",
                               id_name="id",
                               delimiter=",",
                               label_name="y",
                               label_type="float32",
                               dtype="float32")

reader_0.hosts[[0, 1]].component_param(path="examples/data/breast_hetero_host.csv",
                                       format="csv",
                                       id_name="id",
                                       delimiter=",",
                                       label_name=None,
                                       dtype="float32")

intersection_0 = Intersection(name="intersection_0",
                              method="raw",
                              input_data=reader_0.outputs["output_data"])

feature_scale_0 = FeatureScale(name="feature_scale_0",
                               method="standard",
                               train_data=intersection_0.outputs["output_data"])

pipeline.add_task(reader_0)
pipeline.add_task(intersection_0)
pipeline.add_task(feature_scale_0)

pipeline.conf.set("task_parallelism", 1)
pipeline.compile()
pipeline.fit()
print(pipeline.get_dag())
pipeline.deploy([intersection_0, feature_scale_0])


predict_pipeline = StandalonePipeline()
reader_1 = Reader(name="reader_1")
reader_1.guest.component_param(path="examples/data/breast_hetero_guest.csv",
                               format="csv",
                               id_name="id",
                               delimiter=",",
                               label_name="y",
                               label_type="float32",
                               dtype="float32")

reader_1.hosts[[0, 1]].component_param(path="examples/data/breast_hetero_host.csv",
                                       format="csv",
                                       id_name="id",
                                       delimiter=",",
                                       label_name=None,
                                       dtype="float32")


deployed_pipeline = pipeline.get_deployed_pipeline()
deployed_pipeline.intersection_0.input_data = reader_1.outputs["output_data"]

predict_pipeline.add_task(deployed_pipeline)
predict_pipeline.add_task(reader_1)

print("\n\n\n")
print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()
