#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataIO
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model


def main():
    # parties config
    guest = 9999
    host = 10000
    arbiter = 10000
    # 0 for eggroll, 1 for spark
    backend = Backend.EGGROLL
    # 0 for standalone, 1 for cluster
    work_mode = WorkMode.STANDALONE
    # use the work mode below for cluster deployment
    # work_mode = WorkMode.CLUSTER

    # specify input data name & namespace in database
    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role="guest", party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role="guest", party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role="host", party_id=host).algorithm_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role="guest", party_id=guest).algorithm_param(table=guest_eval_data)
    reader_1.get_party_instance(role="host", party_id=host).algorithm_param(table=host_eval_data)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")
    dataio_1 = DataIO(name="dataio_1")

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role="guest", party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role="host", party_id=host).algorithm_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    # define HeteroLR component
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10,
                           early_stopping_rounds=2, validation_freqs=2)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    # set dataio_1 to replicate model from dataio_0
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
    # set train & validate data of hetero_lr_0 component
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data, validate_data=intersection_1.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print (pipeline.get_component("hetero_lr_0").get_summary())


    # predict
    # deploy required components
    pipeline.deploy_component([dataio_0, intersection_0, hetero_lr_0])

    # initiate predict pipeline
    predict_pipeline = PipeLine()

    reader_2 = Reader(name="reader_2")
    reader_2.get_party_instance(role="guest", party_id=guest).algorithm_param(table=guest_eval_data)
    reader_2.get_party_instance(role="host", party_id=host).algorithm_param(table=host_eval_data)
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_2)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.dataio_0.input.data: reader_2.output.data}))
    # run predict model
    predict_pipeline.predict(backend=backend, work_mode=work_mode)


if __name__ == "__main__":
    main()
