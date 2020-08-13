from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.intersection import Intersection
from pipeline.component.union import Union
from pipeline.interface.data import Data
from pipeline.interface.model import Model
from pipeline.component.reader import Reader

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = [{"name": "breast_hetero_host", "namespace": "experiment"},
                   {"name": "breast_hetero_host", "namespace": "experiment"},
                   {"name": "breast_hetero_host", "namespace": "experiment"}]

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=hosts, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_0.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_1.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_1.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

dataio_0 = DataIO(name="dataio_0")
dataio_1 = DataIO(name="dataio_1")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_1.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
intersect_1 = Intersection(name="intersection_1")

union_0 = Union(name="union_0")
hetero_lr_0 = HeteroLR(name="hetero_lr_0", max_iter=20, early_stop="weight_diff")

pipeline.add_component(reader_0)
pipeline.add_component(reader_1)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))
pipeline.add_component(union_0, data=Data(data=[intersect_0.output.data, intersect_1.output.data]))
pipeline.add_component(hetero_lr_0, data=Data(train_data=union_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print(pipeline.get_component("union_0").get_summary())
print(pipeline.get_component("hetero_lr_0").get_summary())

