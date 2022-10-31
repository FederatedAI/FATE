from pipeline import FateStandalonePipeline as Pipeline
from pipeline.components import HeteroLR, DataInput


data_guest = "file://$fate_project_base_abspath/examples/data/breast_hetero_guest.csv"
data_host = "file://$fate_project_base_abspath/examples/data/breast_hetero_host.csv"


pipeline = Pipeline().set_leader("guest", 9999).set_roles(guest=9999, host=10000, arbiter=10001)
data_input = DataInput(name="local_dataset")
data_input.guest[0].component_param(data=data_guest)
data_input.host[0].component_param(data=data_host)
data_input.arbiter[0].component_param(data=data_host)
hetero_lr_0 = HeteroLR(name="hetero_lr_0", max_iter=1)

hetero_lr_0.guest[0].component_param(tol=1e-3)
hetero_lr_0.host[0].component_param(tol=1e-3)
# hetero_lr_0.host[1].component_param(lr=0.2)

# hetero_lr_1 = HeteroLR(name="hetero_lr_1", max_iter=1)

pipeline.add_component(data_input)
pipeline.add_component(hetero_lr_0, train_data=data_input.output.data)
# pipeline.add_component(hetero_lr_1, model=hetero_lr_0.output.model)
pipeline.compile()
pipeline.fit()
