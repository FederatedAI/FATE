from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_linr import HeteroLinR
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "motor_hetero_guest", "namespace": "experiment"}
host_train_data = [{"name": "motor_hetero_host", "namespace": "experiment"},
                   {"name": "motor_hetero_host", "namespace": "experiment"},
                   {"name": "motor_hetero_host", "namespace": "experiment"}]

input_0 = Input(name="train_data")
print ("get input_0's init name {}".format(input_0.name))

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=hosts, arbiter=arbiter)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
hetero_linr_0 = HeteroLinR(name="hetero_linr_0", early_stop="weight_diff", max_iter=20)

print ("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_linr_0, data=Data(train_data=intersect_0.output.data))

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

# print (pipeline.get_component("hetero_linr_0").get_model_param())
print (pipeline.get_component("hetero_linr_0").get_summary())


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


