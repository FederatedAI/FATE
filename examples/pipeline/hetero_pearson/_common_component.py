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
from pipeline.component import HeteroPearson
from pipeline.component.dataio import DataIO
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.utils.tools import load_job_config


# noinspection PyPep8Naming
class dataset(object):
    breast = {
        "guest": {"name": "breast_hetero_guest", "namespace": "experiment"},
        "host": [
            {"name": "breast_hetero_host", "namespace": "experiment"}
        ]
    }


def run_pearson_pipeline(config, namespace, data, common_param=None, guest_only_param=None, host_only_param=None):
    if isinstance(config, str):
        config = load_job_config(config)
    guest_data = data["guest"]
    host_data = data["host"][0]

    guest_data["namespace"] = f"{guest_data['namespace']}{namespace}"
    host_data["namespace"] = f"{host_data['namespace']}{namespace}"

    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=config.parties.guest[0]) \
        .set_roles(guest=config.parties.guest[0], host=config.parties.host[0])

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.parties.guest[0]).algorithm_param(table=guest_data)
    reader_0.get_party_instance(role='host', party_id=config.parties.host[0]).algorithm_param(table=host_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=config.parties.guest[0]) \
        .algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=config.parties.host[0]).algorithm_param(with_label=False)

    intersect_0 = Intersection(name="intersection_0")

    if common_param is None:
        common_param = {}
    hetero_pearson_component = HeteroPearson(name="hetero_pearson_0", **common_param)

    if guest_only_param:
        hetero_pearson_component.get_party_instance("guest", config.parties.guest[0]).algorithm_param(**guest_only_param)

    if host_only_param:
        hetero_pearson_component.get_party_instance("host", config.parties.host[0]).algorithm_param(**host_only_param)

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_pearson_component, data=Data(train_data=intersect_0.output.data))

    pipeline.compile()
    pipeline.fit(backend=config.backend, work_mode=config.work_mode)
    return pipeline


def runner(main_func):
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main_func(args.config)
    else:
        main_func()
