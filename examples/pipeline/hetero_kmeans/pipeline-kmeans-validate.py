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
from pipeline.component.hetero_kmeans import HeteroKmeans
from pipeline.component.intersection import Intersection
from pipeline.component.evaluation import Evaluation
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=host).algorithm_param(table=host_eval_data)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0
    dataio_1 = DataIO(name="dataio_1")

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    param = {
        "k": 3,
        "max_iter": 10
    }

    hetero_kmeans_0 = HeteroKmeans(name='hetero_kmeans_0', **param)
    hetero_kmeans_1 = HeteroKmeans(name='hetero_kmeans_1')
    evaluation_0 = Evaluation(name='evaluation_0', eval_type='clustering')
    evaluation_1 = Evaluation(name='evaluation_1', eval_type='clustering')

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
    # set train & validate data of hetero_lr_0 component

    pipeline.add_component(hetero_kmeans_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_kmeans_1, data=Data(train_data=intersection_1.output.data))
    # print(f"data: {hetero_kmeans_0.output.data.data[0]}")
    pipeline.add_component(evaluation_0, data=Data(data=hetero_kmeans_0.output.data.data[0]))
    pipeline.add_component(evaluation_1, data=Data(data=hetero_kmeans_1.output.data.data[0]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
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
