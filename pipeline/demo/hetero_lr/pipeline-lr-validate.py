from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.input import Input
from pipeline.component.reader import Reader
from pipeline.component.intersection import Intersection
from pipeline.interface.data import Data
from pipeline.interface.model import Model
from pprint import pprint

guest = 9999
hosts = 9999
arbiter = 9999

guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
host_train_data = [{"name": "breast_hetero_host", "namespace": "experiment"}]

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=9999).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=9999).algorithm_param(table=host_train_data[0])

pipeline = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=hosts, arbiter=arbiter)
dataio_0 = DataIO(name="dataio_0")

dataio_0.get_party_instance(role='guest', party_id=9999).algorithm_param(with_label=True, output_format="dense")
dataio_0.get_party_instance(role='host', party_id=9999).algorithm_param(with_label=False)

dataio_1 = DataIO(name="dataio_1")

intersect_0 = Intersection(name="intersection_0")
intersect_1 = Intersection(name="intersection_1")
hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=1)
hetero_lr_1 = HeteroLR(name="hetero_lr_1", early_stop="weight_diff")

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_lr_0, data=Data(train_data=intersect_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.CLUSTER)

# print(pipeline.get_component("intersection_0").get_output_data())
# print(pipeline.get_component("dataio_0").get_model_param())
# print(pipeline.get_component("hetero_lr_0").get_model_param())


# predict

pipeline.deploy_component([dataio_0, hetero_lr_0])

pprint (pipeline._predict_dsl)
pprint (pipeline.get_input_reader_placeholder())

predict_pipeline = PipeLine()
predict_pipeline.add_component(reader_0)
predict_pipeline.add_component(pipeline, data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}))
predict_pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.CLUSTER)

# with open("output.pkl", "wb") as fout:
#     fout.write(pipeline.dump())
