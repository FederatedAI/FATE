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
from pathlib import Path

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.demo.util.demo_util import Config
from pipeline.interface.data import Data


def run_pipeline(config, guest_data, host_data, hetero_pearson, namespace):
    arbiter = config.arbiter
    guest_data["namespace"] = f"{guest_data['namespace']}{namespace}"
    host_data["namespace"] = f"{host_data['namespace']}{namespace}"

    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=config.guest) \
        .set_roles(guest=config.guest, host=config.host[0], arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.guest).algorithm_param(table=guest_data)
    reader_0.get_party_instance(role='host', party_id=config.host[0]).algorithm_param(table=host_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=config.guest).algorithm_param(with_label=True,
                                                                                     output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=config.host[0]).algorithm_param(with_label=False)

    intersect_0 = Intersection(name="intersection_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_pearson, data=Data(train_data=intersect_0.output.data))

    pipeline.compile()
    pipeline.fit(backend=config.backend, work_mode=config.work_mode)
    return pipeline
