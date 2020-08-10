from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_fast_secureboost import HeteroFastSecureBoost
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data

guest = 9999
host = 10000

# prepare

guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = [{"name": "breast_hetero_host", "namespace": "experiment"}]

input_0 = Input(name="train_data")
print("get input_0's init name {}".format(input_0.name))

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
hetero_fast_secure_boost_0 = HeteroFastSecureBoost(name="hetero_fast_secure_boost_0",
                                                   num_trees=4,
                                                   task_type='classification',
                                                   objective_param={"objective": "cross_entropy"},
                                                   encrypt_param={"method": "iterativeAffine"},
                                                   validation_freqs=1,
                                                   work_mode='layered',
                                                   guest_depth=2,
                                                   host_depth=3
                                                   )

print("get input_0's name {}".format(input_0.name))
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_fast_secure_boost_0, data=Data(train_data=intersect_0.output.data))

# fitting

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                           {"guest": {9999: guest_train_data},
                            "host": {
                              10000: host_train_data[0]
                             }
                            }

                       })

print(pipeline.get_component("intersection_0").get_output_data())
print(pipeline.get_component("dataio_0").get_model_param())
print(pipeline.get_component("hetero_fast_secure_boost_0").get_summary())


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