from pipeline.components.fate import HeteroLR
from pipeline.components.fate import Reader
from pipeline.components.fate import FeatureScale
from pipeline.components.fate import Intersection
from pipeline.pipeline import FateFlowPipeline


pipeline = FateFlowPipeline().set_scheduler_party_id(party_id=10001).set_roles(
        guest=9999, host=[10000, 10001], arbiter=10001)
reader_0 = Reader(name="reader_0")
reader_0.guest.component_param(path="file:///Users/tonly/FATE_1/examples/data/breast_hetero_guest.csv",
                               format="csv",
                               id_name="id",
                               delimiter=",",
                               label_name="y",
                               label_type="float32",
                               dtype="float32")

reader_0.hosts[[0, 1]].component_param(path="file:///Users/tonly/FATE_1/examples/data/breast_hetero_host.csv",
                                       format="csv",
                                       id_name="id",
                                       delimiter=",",
                                       label_name=None,
                                       dtype="float32")

intersection_0 = Intersection(name="intersection_0",
                              method="raw",
                              input_data=reader_0.outputs["output_data"])

intersection_1 = Intersection(name="intersection_1",
                              method="raw",
                              input_data=reader_0.outputs["output_data"])

lr_0 = HeteroLR(name="lr_0",
                validate_data=intersection_0.outputs["output_data"],
                train_data=intersection_0.outputs["output_data"],
                max_iter=1,
                learning_rate=0.01,
                batch_size=569)

lr_1 = HeteroLR(name="lr_1",
                validate_data=intersection_1.outputs["output_data"],
                test_data=intersection_1.outputs["output_data"],
                input_model=lr_0.outputs["output_model"])

pipeline.add_task(reader_0)
pipeline.add_task(intersection_0)
pipeline.add_task(intersection_1)
pipeline.add_task(lr_0)
pipeline.compile()
print (pipeline.get_dag())
pipeline.fit()
