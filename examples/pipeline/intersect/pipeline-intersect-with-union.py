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
from pipeline.component import Intersection
from pipeline.component import Union
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    # specify input data name & namespace in database
    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    param = {
        "intersect_method": "raw",
        "sync_intersect_ids": True,
        "only_output_key": True,
        "raw_params": {
            "use_hash": True,
            "hash_method": "sha256",
            "salt": "12345",
            "base64": True,
            "join_role": "host"
        }
    }
    # define Intersection components
    intersections = []
    for i in range(200):
        intersection_tmp = Intersection(name="intersection_" + str(i), **param)
        intersections.append(intersection_tmp)

    union_0 = Union(name="union_0")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    # set data input sources of intersection components
    for i in range(len(intersections)):
        pipeline.add_component(intersections[i], data=Data(data=reader_0.output.data))

    # set data output of intersection components
    intersection_outputs = [intersection_tmp.output.data for intersection_tmp in intersections]
    pipeline.add_component(union_0, data=Data(data=intersection_outputs))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
