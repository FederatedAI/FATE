from pipeline.component.homo_data_split import HomoDataSplit
from pipeline.component.reader import Reader

from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.interface.data import Data

guest = 9999
host = 10000

guest_train_data = {"name": "breast_homo_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_homo_host", "namespace": "experiment"}

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense",
                                                                          label_name="y", label_type="int")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=True)

homo_data_split_0 = HomoDataSplit(name="homo_data_split_0", stratified=True, test_size=0.3, validate_size=0.2)

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(homo_data_split_0, data=Data(data=dataio_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print(pipeline.get_component("dataio_0").get_model_param())
print(pipeline.get_component("homo_data_split_0").get_summary())
