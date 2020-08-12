from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_linr import HeteroLinR
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model

guest = 9999
host = 10000
arbiter = 10002

guest_train_data = [{"name": "motor_hetero_guest", "namespace": "experiment"},
                    {"name": "motor_hetero_guest", "namespace": "experiment"}]
host_train_data = [{"name": "motor_hetero_host", "namespace": "experiment"},
                   {"name": "motor_hetero_host", "namespace": "experiment"}]

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data[0])
reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data[0])

reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data[1])
reader_1.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data[1])


dataio_0 = DataIO(name="dataio_0")
dataio_1 = DataIO(name="dataio_1")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_1.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersection_0 = Intersection(name="intersection_0")
intersect_1 = Intersection(name="intersection_1")

hetero_linr_0 = HeteroLinR(name="hetero_linr_0", penalty="L2", optimizer="sgd", tol=0.001,
                           alpha=0.01, max_iter=20, early_stop="weight_diff", batch_size=-1,
                           learning_rate=0.15, decay=0.0, decay_sqrt=False,
                           init_param={"init_method": "zeros"},
                           encrypted_mode_calculator_param={"mode": "fast"},
                           validation_freqs=1, early_stopping_rounds=5,
                           metrics=["mean_absolute_error", "root_mean_squared_error"],
                           use_first_metric_only=False)

pipeline.add_component(reader_0)
pipeline.add_component(reader_1)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))

pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))

pipeline.add_component(hetero_linr_0, data=Data(train_data=intersection_0.output.data,
                                                validate_data=intersect_1.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print (pipeline.get_component("hetero_linr_0").get_model_param())
print (pipeline.get_component("hetero_linr_0").get_summary())