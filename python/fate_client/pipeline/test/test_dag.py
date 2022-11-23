from pipeline.components.fate import HeteroLR
from pipeline.components.fate import Reader
from pipeline.pipeline import StandalonePipeline


pipeline = StandalonePipeline().set_scheduler_party_id(party_id=10001).set_roles(
        guest=9999, host=[10000, 10001], arbiter=10001)
reader_0 = Reader(name="reader_0")
reader_0.guest.component_param(path="examples/data/breast_hetero_guest.csv")
reader_0.hosts[[0, 1]].component_param(path="examples/data/breast_hetero_host.csv")
lr_0 = HeteroLR(name="lr_0", train_data=reader_0.outputs["data"], max_iter=1)
lr_1 = HeteroLR(name="lr_1", test_data=reader_0.outputs["data"], input_model=lr_0.outputs["output_model"])
pipeline.add_component(reader_0)
pipeline.add_component(lr_0)
pipeline.add_component(lr_1)
pipeline.conf.set("task_parallelism", 1)
pipeline.compile()
print(pipeline.get_dag())
