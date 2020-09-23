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
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.utils.tools import load_job_config


# noinspection PyPep8Naming
class dataset(object):
    breast = {
        "guest": {"name": "breast_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "breast_homo_host", "namespace": "experiment"},
            {"name": "breast_homo_host", "namespace": "experiment"}
        ]
    }
    vehicle = {
        "guest": {"name": "vehicle_scale_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"},
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"}
        ]
    }


def run_homo_nn_pipeline(config, namespace, data: dict, nn_component, num_host):
    if isinstance(config, str):
        config = load_job_config(config)

    guest_train_data = data["guest"]
    host_train_data = data["host"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    hosts = config.parties.host[:num_host]
    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=config.parties.guest[0]) \
        .set_roles(guest=config.parties.guest[0], host=hosts, arbiter=config.parties.arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.parties.guest[0]).algorithm_param(table=guest_train_data)
    for i in range(num_host):
        reader_0.get_party_instance(role='host', party_id=hosts[i]) \
            .algorithm_param(table=host_train_data[i])

    dataio_0 = DataIO(name="dataio_0", with_label=True)
    dataio_0.get_party_instance(role='guest', party_id=config.parties.guest[0]) \
        .algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=True)

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(nn_component, data=Data(train_data=dataio_0.output.data))
    pipeline.compile()
    pipeline.fit(backend=config.backend, work_mode=config.work_mode)
    print(pipeline.get_component("homo_nn_0").get_summary())
    pipeline.deploy_component([dataio_0, nn_component])

    # predict
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.dataio_0.input.data: reader_0.output.data}))
    # run predict model
    predict_pipeline.predict(backend=config.backend, work_mode=config.work_mode)
    pipeline.get_train_conf()
    pipeline.get_train_dsl()


def runner(main_func):
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main_func(args.config)
    else:
        main_func()
