from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.union import Union
from pipeline.interface.data import Data
from pipeline.component.reader import Reader

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "tag_value_1", "namespace": "experiment"}
host_train_data = [{"name": "tag_value_1", "namespace": "experiment"},
                   {"name": "tag_value_2", "namespace": "experiment"}]

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=hosts, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_0.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_1.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_1.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

dataio_0 = DataIO(name="dataio_0", input_format="tag", with_label=False, tag_with_value=True,
                  delimitor=",", output_format="dense")
union_0 = Union(name="union_0", allow_missing=False, keep_duplicate=True)

pipeline.add_component(reader_0)
pipeline.add_component(reader_1)
pipeline.add_component(union_0, data=Data(data=[reader_0.output.data, reader_1.output.data]))
pipeline.add_component(dataio_0, data=Data(data=union_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)


print(pipeline.get_component("union_0").get_summary())

