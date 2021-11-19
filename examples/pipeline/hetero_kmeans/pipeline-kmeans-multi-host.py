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
from pipeline.component import HeteroKmeans
from pipeline.component import Intersection
from pipeline.component import Evaluation
from pipeline.component import Reader
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = [{"name": "breast_hetero_host", "namespace": f"experiment{namespace}"},
                       {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}]

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
    reader_0.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_train_data[0])
    reader_0.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_train_data[1])

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts[0]).component_param(with_label=False)
    data_transform_0.get_party_instance(role='host', party_id=hosts[1]).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")

    param = {
        "k": 3,
        "max_iter": 10
    }

    hetero_kmeans_0 = HeteroKmeans(name='hetero_kmeans_0', **param)
    evaluation_0 = Evaluation(name='evaluation_0', eval_type='clustering')

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))

    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    pipeline.add_component(hetero_kmeans_0,
                           data=Data(train_data=intersection_0.output.data))
    # print(f"data: {hetero_kmeans_0.output.data.data[0]}")
    pipeline.add_component(evaluation_0, data=Data(data=hetero_kmeans_0.output.data.data[0]))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    print(pipeline.get_component("hetero_kmeans_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
