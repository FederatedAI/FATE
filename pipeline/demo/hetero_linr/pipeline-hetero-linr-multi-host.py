from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_linr import HeteroLinR
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

guest = 9999
hosts = [10000, 10001]
arbiter = 10002

guest_train_data = {"name": "motor_hetero_guest", "namespace": "experiment"}
host_train_data = [{"name": "motor_hetero_host", "namespace": "experiment"},
                   {"name": "motor_hetero_host", "namespace": "experiment"}]


pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=hosts, arbiter=arbiter)

reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
reader_0.get_party_instance(role='host', party_id=hosts[0]).algorithm_param(table=host_train_data[0])
reader_0.get_party_instance(role='host', party_id=hosts[1]).algorithm_param(table=host_train_data[1])

dataio_0 = DataIO(name="dataio_0")
dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, label_name="motor_speed",
                                                                         label_type="float", output_format="dense")
dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=False)

intersection_0 = Intersection(name="intersection_0")
hetero_linr_0 = HeteroLinR(name="hetero_linr_0", penalty="L2", optimizer="sgd", tol=0.001,
                           alpha=0.01, max_iter=10, early_stop="weight_diff", batch_size=-1,
                           learning_rate=0.15, decay=0.0, decay_sqrt=False,
                           init_param={"init_method": "zeros"},
                           encrypted_mode_calculator_param={"mode": "fast"})

pipeline.add_component(reader_0)
pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
pipeline.add_component(hetero_linr_0, data=Data(train_data=intersection_0.output.data))

pipeline.compile()

pipeline.fit(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE)

print (pipeline.get_component("hetero_linr_0").get_summary())


# predict
# deploy required components
pipeline.deploy_component([dataio_0, hetero_linr_0])

predict_pipeline = PipeLine()
# add data reader onto predict pipeline
predict_pipeline.add_component(reader_0)
# add selected components from train pipeline onto predict pipeline
# specify data source
predict_pipeline.add_component(pipeline,
                               data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}))
# run predict model
predict_pipeline.predict(backend=backend, work_mode=work_mode)
