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
from pipeline.component import ColumnExpand
from pipeline.component import DataIO
from pipeline.component import Reader
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

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role="guest", party_id=guest).set_roles(guest=guest)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)

    # define ColumnExpand components
    column_expand_0 = ColumnExpand(name="column_expand_0")
    column_expand_0.get_party_instance(role="guest", party_id=guest).component_param(need_run=True,
                                                                                     method="manual",
                                                                                     append_header=["x_0", "x_1", "x_2", "x_3"],
                                                                                     fill_value=[0, 0.2, 0.5, 1])
    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0
    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role="guest", party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.component_param(with_label=True, output_format="dense")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(column_expand_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_0, data=Data(data=column_expand_0.output.data))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)
    import json
    print(f"{json.dumps(pipeline.get_train_conf(), indent=4)}")
    print(f"\n{json.dumps(pipeline.get_train_dsl(), indent=4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
