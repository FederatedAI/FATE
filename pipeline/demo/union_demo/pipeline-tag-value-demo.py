from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.input import Input
from pipeline.component.union import Union
from pipeline.interface.data import Data

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "tag_value_1", "namespace": "experiment"}
host_train_data = [{"name": "tag_value_1", "namespace": "experiment"},
                   {"name": "tag_value_2", "namespace": "experiment"}]

input_0 = Input(name="train_data_0")
input_1 = Input(name="train_data_1")

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=hosts, arbiter=arbiter)

dataio_0 = DataIO(name="dataio_0", input_format="tag", with_label=False, tag_with_value=True,
                  delimiter=",", output_format="dense")
union_0 = Union(name="union_0", allow_missing=False, keep_duplicate=True)

pipeline.add_component(union_0, data=Data(data=[input_0.data, input_1.data]))
pipeline.add_component(dataio_0, data=Data(data=union_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
             feed_dict={input_0:
                           {"guest":
                                {9999: guest_train_data},
                            "host": {
                                10000: host_train_data[0],
                                10001: host_train_data[1]
                                }
                            },
                        input_1:
                            {"guest":
                                {9999: guest_train_data},
                            "host": {
                                10000: host_train_data[1],
                                10001: host_train_data[0]
                                }
                            },

                       })


print(pipeline.get_component("union_0").get_summary())

