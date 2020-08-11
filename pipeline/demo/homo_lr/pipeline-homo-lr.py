from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.homo_lr import HomoLR
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

guest = 9999
host = 10000
arbiter = 10002

guest_train_data = {"name": "breast_homo_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_homo_host", "namespace": "experiment"}

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=True)

homo_lr_0 = HomoLR(name="homo_lr_0", max_iter=20)

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(homo_lr_0, data=Data(data=dataio_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

# predict

pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print (pipeline.get_component("dataio_0").get_model_param())
print (pipeline.get_component("homo_lr_0").get_summary())

with open("output.pkl", "wb") as fout:
    fout.write(pipeline.dump())
