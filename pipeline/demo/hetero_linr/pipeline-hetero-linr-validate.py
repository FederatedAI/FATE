from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_linr import HeteroLinR
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data
from pipeline.interface.model import Model


guest = 9999
host = 10000
arbiter = 10002

guest_train_data = [{"name": "motor_hetero_guest", "namespace": "experiment"},
                    {"name": "motor_hetero_guest", "namespace": "experiment"}]
host_train_data = [{"name": "motor_hetero_host", "namespace": "experiment"},
                   {"name": "motor_hetero_host", "namespace": "experiment"}]

input_0 = Input(name="train_data")
input_1 = Input(name="eval_data")

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

dataio_0 = DataIO(name="dataio_0")
dataio_1 = DataIO(name="dataio_1")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_1.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
intersect_1 = Intersection(name="intersection_1")

hetero_linr_0 = HeteroLinR(name="hetero_linr_0", early_stop="weight_diff", max_iter=20, learning_rate=0.15,
                           validation_freqs=1, early_stopping_rounds=3)

print ("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(dataio_1, data=Data(data=input_1.data), model=Model(dataio_0.output.model))

pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))

pipeline.add_component(hetero_linr_0, data=Data(train_data=intersect_0.output.data,
                                                validate_data=intersect_1.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                           {"guest": {9999: guest_train_data[0]},
                            "host": {10000: host_train_data[0]}
                            },
                        input_1:
                            {"guest": {9999: guest_train_data[1]},
                            "host": {10000: host_train_data[1]}
                            },

                       })

print (pipeline.get_component("hetero_linr_0").get_model_param())
print (pipeline.get_component("hetero_linr_0").get_summary())