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

import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_statistics import DataStatistics
from pipeline.component.dataio import DataIO
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.demo.util.demo_util import Config
from pipeline.interface.data import Data
from pipeline.interface.model import Model


def main(config="../config.yaml"):
    # obtain config
    config = Config(config)
    guest = config.guest
    host = config.host[0]
    arbiter = config.arbiter
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).\
        set_roles(guest=guest, host=host, arbiter=arbiter)
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")

    dataio_0.get_party_instance(role='guest', party_id=guest).\
        algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    dataio_1 = DataIO(name="dataio_1")

    intersect_0 = Intersection(name="intersection_0")
    intersect_1 = Intersection(name="intersection_1")
    statics_0 = DataStatistics(name="data_statistic_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, model=Model(model=dataio_0.output.model), data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersect_1, data=Data(data=dataio_1.output.data))
    pipeline.add_component(statics_0, data=Data(data=intersect_0.output.data))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    print(pipeline.get_component("intersection_0").get_output_data())
    print(pipeline.get_component("dataio_0").get_model_param())
    print(pipeline.get_component("data_statistic_0").get_model_param())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
