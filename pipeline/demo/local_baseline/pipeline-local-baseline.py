from pipeline.component.reader import Reader
from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.evaluation import Evaluation
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.intersection import Intersection
from pipeline.component.local_baseline import LocalBaseline
from pipeline.interface.data import Data

guest = 9999
host = 10000
arbiter = 10002

guest_train_data = {"name": "vehicle_scale_hetero_guest", "namespace": "experiment"}
host_train_data = {"name": "vehicle_scale_hetero_host", "namespace": "experiment"}

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense",
                                                                          label_type="int", label_name="y")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersection_0 = Intersection(name="intersection_0", intersect_method="rsa", sync_intersect_ids=True,
                              only_output_key=False)
hetero_lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="nesterov_momentum_sgd",
                       tol=0.0001, alpha=0.0001, max_iter=30, batch_size=-1,
                       early_stop="diff", learning_rate=0.15, init_param={"init_method": "zeros"})

local_baseline_0 = LocalBaseline(name="local_baseline_0", model_name="LogisticRegression",
                                 model_opts={"penalty": "l2", "tol": 0.0001, "C": 1.0, "fit_intercept": True,
                                             "solver": "lbfgs", "max_iter": 5, "multi_class": "ovr"})
local_baseline_0.get_party_instance(role='guest', party_id=guest).algorithm_param(need_run=True)
local_baseline_0.get_party_instance(role='host', party_id=host).algorithm_param(need_run=False)

evaluation_0 = Evaluation(name="evaluation_0", eval_type="multi", pos_label=1)
evaluation_0.get_party_instance(role='guest', party_id=guest).algorithm_param(need_run=True)
evaluation_0.get_party_instance(role='host', party_id=host).algorithm_param(need_run=False)

evaluation_1 = Evaluation(name="evaluation_1", eval_type="multi", pos_label=1)
evaluation_1.get_party_instance(role='guest', party_id=guest).algorithm_param(need_run=True)
evaluation_1.get_party_instance(role='host', party_id=host).algorithm_param(need_run=False)

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
pipeline.add_component(local_baseline_0, data=Data(train_data=intersection_0.output.data))
pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))
pipeline.add_component(evaluation_1, data=Data(data=local_baseline_0.output.data))

# pipeline.set_deploy_end_component([dataio_0])
# pipeline.deploy_component([dataio_0])

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print(pipeline.get_component("intersection_0").get_output_data())
print(pipeline.get_component("dataio_0").get_model_param())
print(pipeline.get_component("hetero_lr_0").get_model_param())
print()
print(pipeline.get_component("local_baseline_0").get_model_param())
print(pipeline.get_component("local_baseline_0").get_summary())
