from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_ftl import HeteroFTL
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data
from pipeline.interface.model import Model

guest = 9999
host = 10000

guest_train_data = {"name": "nus_wide_guest", "namespace": "hetero"}
host_train_data = [{"name": "nus_wide_host", "namespace": "hetero"}]

input_0 = Input(name="train_data")
print("get input_0's init name {}".format(input_0.name))

pipeline = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=host)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

hetero_ftl_0 = HeteroFTL(name='hetero_ftl_0', epochs=10, alpha=1, batch_size=-1)
hetero_ftl_0.add_nn_layer(Dense(units=32, activation='sigmoid'))
hetero_ftl_0.compile(optimizer=optimizers.Adam(lr=0.01))

print("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(hetero_ftl_0, data=Data(train_data=Data(data=dataio_0.output.data)))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                            {"guest": {9999: guest_train_data},
                             "host": {
                                 10000: host_train_data[0]
                             }
                             }

                        })

# predict

pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
                 feed_dict={input_0:
                                {"guest":
                                     {9999: guest_train_data},
                                 "host": {
                                     10000: host_train_data[0]
                                 }
                                 }
                            })

with open("output.pkl", "wb") as fout:
    fout.write(pipeline.dump())