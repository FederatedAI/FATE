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
from pipeline.component import HeteroLR
from pipeline.component import Intersection
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
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment_sid{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment_sid{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).\
        set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0", with_match_id=True)
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")
    lr_param = {
        "name": "hetero_lr_0",
        "penalty": "L2",
        "optimizer": "rmsprop",
        "tol": 0.0001,
        "alpha": 0.01,
        "max_iter": 30,
        "early_stop": "diff",
        "batch_size": 320,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "zeros"
        },
        "sqn_param": {
            "update_interval_L": 3,
            "memory_M": 5,
            "sample_size": 5000,
            "random_seed": None
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        }
    }

    hetero_lr_0 = HeteroLR(**lr_param)
    # evaluation_0 = Evaluation(name='evaluation_0')

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    # pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

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
