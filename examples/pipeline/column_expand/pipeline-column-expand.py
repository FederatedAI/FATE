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
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role="guest", party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role="guest", party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role="host", party_id=host).algorithm_param(table=host_train_data)

    # define ColumnExpand components
    column_expand_0 = ColumnExpand(name="column_expand_0")
    column_expand_0.get_party_instance(role="guest", party_id=guest).algorithm_param(need_run=True,
                                                                                     method="manual",
                                                                                     append_header=["x_0", "x_1", "x_2", "x_3"],
                                                                                     fill_value=[0, 0.2, 0.5, 1])
    column_expand_0.get_party_instance(role="host", party_id=host).algorithm_param(need_run=False)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role="guest", party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role="host", party_id=host).algorithm_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0", intersect_method="rsa",
                                  sync_intersect_ids=True, only_output_key=False)

    param = {
        "penalty": "L2",
        "optimizer": "nesterov_momentum_sgd",
        "tol": 0.0001,
        "alpha": 0.01,
        "max_iter": 20,
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

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(column_expand_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_0, data=Data(data=column_expand_0.output.data))
    # set data input sources of intersection components
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    # set train & validate data of hetero_lr_0 component

    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("hetero_lr_0").get_summary())

    # predict
    # deploy required components
    pipeline.deploy_component([column_expand_0, dataio_0, intersection_0, hetero_lr_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_0)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(pipeline,
                                   data=Data(predict_input={pipeline.column_expand_0.input.data: reader_0.output.data}))
    # run predict model
    predict_pipeline.predict(backend=backend, work_mode=work_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
