from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.homo_nn import HomoNN
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "breast_hetero_guest", "namespace": "hetero"}
host_train_data = [{"name": "breast_hetero_guest", "namespace": "hetero"},
                   {"name": "breast_hetero_guest", "namespace": "hetero"},
                   {"name": "breast_hetero_guest", "namespace": "hetero"}]

pipeline = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=hosts, arbiter=arbiter)

reader_0 = Reader(name="reader_1")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_0.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

dataio_0 = DataIO(name="dataio_0", with_label=True)

dataio_0.get_party_instance(role='guest', party_id=9999).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=[10000, 10001]).algorithm_param(with_label=True)

homo_nn_0 = HomoNN(name="homo_nn_0", max_iter=10)
homo_nn_0.add(Dense(units=1, input_shape=(10, )))
homo_nn_0.compile(optimizer=optimizers.SGD(lr=0.1), metrics=["AUC"], loss="binary_crossentropy")

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(homo_nn_0, data=Data(train_data=dataio_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print(pipeline.get_component("homo_nn_0").get_output_data())


# predict
pipeline.deploy_component([dataio_0, homo_nn_0])
pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

# with open("output.pkl", "wb") as fout:
#     fout.write(pipeline.dump())
