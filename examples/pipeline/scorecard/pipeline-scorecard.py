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
from pipeline.component import Scorecard
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

    guest_train_data = {"name": "default_credit_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "default_credit_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role="guest", party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role="guest", party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0", intersect_method="rsa",
                                  sync_intersect_ids=True, only_output_key=False)

    param = {
        "penalty": "L2",
        "optimizer": "nesterov_momentum_sgd",
        "tol": 0.0001,
        "alpha": 0.01,
        "max_iter": 5,
        "early_stop": "weight_diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "random_uniform"
        },
        "sqn_param": {
            "update_interval_L": 3,
            "memory_M": 5,
            "sample_size": 5000,
            "random_seed": None
        }
    }

    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **param)

    # define Scorecard component
    scorecard_0 = Scorecard(name="scorecard_0")
    scorecard_0.get_party_instance(role="guest", party_id=guest).component_param(need_run=True,
                                                                                 method="credit",
                                                                                 offset=500,
                                                                                 factor=20,
                                                                                 factor_base=2,
                                                                                 upper_limit_ratio=3,
                                                                                 lower_limit_value=0)
    scorecard_0.get_party_instance(role="host", party_id=host).component_param(need_run=False)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))

    pipeline.add_component(scorecard_0, data=Data(data=hetero_lr_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit()

    # query component summary
    # print(pipeline.get_component("scorecard_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
