from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.input import Input
from pipeline.component.psi import PSI
from pipeline.interface.data import Data
from pipeline.interface.model import Model

guest = 9999
host = 10000

guest_train_data = {"name": "breast_homo_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_homo_host", "namespace": "experiment"}

input_0 = Input(name="train_data_0")
input_1 = Input(name="train_data_1")

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, )
dataio_0 = DataIO(name="dataio_0")
dataio_1 = DataIO(name="dataio_1")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=False, output_format="dense")
dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=False, output_format="dense")

dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False, output_format="dense")
dataio_1.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False, output_format="dense")

psi_0 = PSI(name='psi_0', max_bin_num=20)

pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(dataio_1, data=Data(data=input_1.data), model=Model(dataio_0.output.model))
pipeline.add_component(psi_0, data=Data(train_data=dataio_0.output.data, validate_data=dataio_1.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={
                 input_0: {
                     "guest": {
                        9999: guest_train_data
                     },
                     "host": {
                        10000: guest_train_data
                     }
                 },
                 input_1: {
                     "guest": {
                        9999: host_train_data
                     },
                     "host": {
                        10000: host_train_data
                     }
                 }
             })
