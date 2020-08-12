from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_pearson import HeteroPearson
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

guest = 9999
host = 10000
arbiter = 10002

guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

dataio_0 = DataIO(name="dataio_0")
dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

intersect_0 = Intersection(name="intersection_0")
hetero_pearson_0 = HeteroPearson(name="hetero_pearson_0", column_indexes=-1)

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_pearson_0, data=Data(train_data=intersect_0.output.data))

# pipeline.set_deploy_end_component([dataio_0])
# pipeline.deploy_component([dataio_0])

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

# print (pipeline.get_component("intersection_0").get_output_data())
# print (pipeline.get_component("dataio_0").get_model_param())
print (pipeline.get_component("hetero_pearson_0").get_model_param())
print (pipeline.get_component("hetero_pearson_0").get_summary())
# pipeline.get_component("intersection_0").summary("intersect_count", "intersect_rate")
