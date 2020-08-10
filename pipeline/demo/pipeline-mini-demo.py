from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.input import Input
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data
from pipeline.interface.model import Model

# define party ids
guest = 10000
host = 9999
arbiter = host

guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

input_0 = Input(name="train_data")
input_1 = Input(name="validate_data")

# initialize pipeline
pipeline = PipeLine()
# set job initiator
pipeline.set_initiator(role='guest', party_id=guest)
# set participants information
pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

# define DataIO components
dataio_0 = DataIO(name="dataio_0") # start component numbering at 0
dataio_1 = DataIO(name="dataio_1")

# get DataIO party instance of guest
dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
# configure DataIO for guest
dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
# get and configure DataIO party instance of host
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

#dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
#dataio_1.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

# define Intersection components
intersection_0 = Intersection(name="intersection_0")
intersection_1 = Intersection(name="intersection_1")

# define HeteroLR component
hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10,
                       early_stopping_rounds=2, validation_freqs=2)

# add components to pipeline, in order of task execution
pipeline.add_component(dataio_0, data=Data(data=input_0.data))
# set dataio_1 to replicate model from dataio_0
pipeline.add_component(dataio_1, data=Data(data=input_1.data), model=Model(dataio_0.output.model_output))
# set data input sources of intersection components
pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
# set train & validate data of hetero_lr_0 component
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data, validate_data=intersection_1.output.data))

# compile pipeline once finished adding modules, this step will form conf and dsl files for running job
pipeline.compile()

# fit model
pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                           {"guest": {guest: guest_train_data},
                            "host": {
                              host: host_train_data
                             }
                            },
                        input_1:
                           {"guest": {guest: guest_train_data},
                            "host": {
                              host: host_train_data
                             }
                            }

                       })

print (pipeline.get_component("hetero_lr_0").get_summary())


# predict with result model of this pipeline

pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
                 feed_dict={input_0:
                                {"guest": {
                                    guest: guest_train_data},
                                 "host": {
                                     host: host_train_data}
                                },
                            input_1:
                                 {"guest": {
                                     guest: guest_train_data},
                                  "host": {
                                     host: host_train_data}
                                 }
                             })

