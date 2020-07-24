from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.homo_nn import HomoNN
from pipeline.component.input import Input
from pipeline.interface.data import Data
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "breast_hetero_guest", "namespace": "hetero"}
host_train_data = [{"name": "breast_hetero_guest", "namespace": "hetero"},
                   {"name": "breast_hetero_guest", "namespace": "hetero"},
                   {"name": "breast_hetero_guest", "namespace": "hetero"}]

input_0 = Input(name="train_data")
print("get input_0's init name {}".format(input_0.name))

pipeline = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=hosts, arbiter=arbiter)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=9999).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=[10000, 10001]).algorithm_param(with_label=False)

homo_nn_0 = HomoNN(name="homo_nn_0", max_iter=10)
homo_nn_0.add(Dense(units=4, input_shape=(20, )))
homo_nn_0.compile(optimizer=optimizers.SGD(lr=0.1), metrics=["AUC"], loss="binary_crossentropy")

print("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(homo_nn_0, data=Data(train_data=dataio_0.output.data))

pipeline.deploy_component([dataio_0, homo_nn_0])

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                            {"guest": {9999: guest_train_data},
                             "host": {
                                 10000: host_train_data[0],
                                 10001: host_train_data[1]
                             }
                             }

                        })

print(pipeline.get_component("homo_nn_0").get_output_data())


# predict

pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
                 feed_dict={input_0:
                                {"guest":
                                     {9999: guest_train_data},
                                 "host": {
                                     10000: host_train_data[0],
                                     10001: host_train_data[1]
                                 }
                                 }
                            })

with open("output.pkl", "wb") as fout:
    fout.write(pipeline.dump())
