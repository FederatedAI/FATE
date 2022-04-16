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
from pipeline.component import DataTransform
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data, Cache
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")

    data_transform_0.get_party_instance(
        role='guest', party_id=guest).component_param(
        with_label=False, output_format="dense")
    data_transform_0.get_party_instance(
        role='host', party_id=host).component_param(
        with_label=False, output_format="dense")

    param_0 = {
        "intersect_method": "rsa",
        "rsa_params": {
            "hash_method": "sha256",
            "final_hash_method": "sha256",
            "key_length": 2048
        },
        "run_cache": True
    }
    param_1 = {
        "intersect_method": "rsa",
        "sync_intersect_ids": False,
        "only_output_key": True,
        "rsa_params": {
            "hash_method": "sha256",
            "final_hash_method": "sha256",
            "key_length": 2048
        }
    }
    intersect_0 = Intersection(name="intersect_0", **param_0)
    intersect_1 = Intersection(name="intersect_1", **param_1)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(
        intersect_1, data=Data(
            data=data_transform_0.output.data), cache=Cache(
            intersect_0.output.cache))

    pipeline.compile()

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
