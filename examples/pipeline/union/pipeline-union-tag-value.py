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
from pipeline.component import DataIO
from pipeline.component import Reader
from pipeline.component import Union
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config
from pipeline.runtime.entity import JobParameters


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = [{"name": "tag_value_1", "namespace": f"experiment{namespace}"},
                        {"name": "tag_value_2", "namespace": f"experiment{namespace}"},
                        {"name": "tag_value_3", "namespace": f"experiment{namespace}"}]

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data[0])

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data[1])

    reader_2 = Reader(name="reader_2")
    reader_2.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data[2])

    union_0 = Union(name="union_0", allow_missing=False, keep_duplicate=True, need_run=True)

    dataio_0 = DataIO(name="dataio_0", input_format="tag", with_label=False, tag_with_value=True,
                      delimitor=",", output_format="dense")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(reader_2)
    pipeline.add_component(union_0, data=Data(data=[reader_0.output.data, reader_1.output.data, reader_2.output.data]))
    pipeline.add_component(dataio_0, data=Data(data=union_0.output.data))

    pipeline.compile()

    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
