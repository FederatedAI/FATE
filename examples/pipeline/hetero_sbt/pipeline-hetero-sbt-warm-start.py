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
import json

from pipeline.backend.pipeline import PipeLine
from pipeline.component import HeteroSecureBoost
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils.tools import load_job_config


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def main(config="../../config.yaml", namespace=""):
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host[0]
    arbiter = parties.arbiter[0]
    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0", output_format='dense')

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True)
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")

    pipeline.add_component(reader_0)

    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))

    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    # secure boost component
    hetero_secure_boost_0 = HeteroSecureBoost(name="hetero_secure_boost_0",
                                              num_trees=3,
                                              task_type="classification",
                                              objective_param={"objective": "cross_entropy"},
                                              encrypt_param={"method": "Paillier"},
                                              tree_param={"max_depth": 3},
                                              validation_freqs=1)

    hetero_secure_boost_1 = HeteroSecureBoost(name="hetero_secure_boost_1",
                                              num_trees=3,
                                              task_type="classification",
                                              objective_param={"objective": "cross_entropy"},
                                              encrypt_param={"method": "Paillier"},
                                              tree_param={"max_depth": 3},
                                              validation_freqs=1)

    pipeline.add_component(hetero_secure_boost_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_secure_boost_1, data=Data(train_data=intersection_0.output.data),
                           model=Model(model=hetero_secure_boost_0.output.model))

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secure_boost_1.output.data))

    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    prettify(pipeline.get_component("hetero_secure_boost_0").get_summary())
    prettify(pipeline.get_component("hetero_secure_boost_1").get_summary())
    prettify(pipeline.get_component("evaluation_0").get_summary())

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
