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
from pipeline.component.data_statistics import DataStatistics
from pipeline.component.dataio import DataIO
from pipeline.component.hetero_feature_selection import HeteroFeatureSelection
from pipeline.component.homo_secureboost import HomoSecureBoost
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    config = Config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_homo_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_homo_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "homo_breast_test", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "homo_breast_test", "namespace": f"experiment{namespace}"}

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
    dataio_0 = DataIO(name="dataio_0", with_label=True, output_format="dense")  # start component numbering at 0
    dataio_1 = DataIO(name="dataio_1")

    statistic_0 = DataStatistics(name='statistic_0')

    param = {
        "task_type": "classification",
        "learning_rate": 0.1,
        "num_trees": 3,
        "subsample_feature_rate": 1,
        "n_iter_no_change": False,
        "tol": 0.0001,
        "bin_num": 50,
        "validation_freqs": 1,
        "tree_param": {
            "max_depth": 3
        },
        "objective_param": {
            "objective": "cross_entropy"
        },
        "predict_param": {
            "threshold": 0.5
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        }
    }

    secureboost_0 = HomoSecureBoost(name='secureboost_0', **param)

    param = {
        "name": 'homo_feature_selection_0',
        "filter_methods": ["manually", "homo_sbt_filter"],
        "manually_param": {
            "filter_out_indexes": [1]
        },
        "sbt_param": {
            "metrics": ["feature_importance"],
            "filter_type": ["threshold"],
            "take_high": [True],
            "threshold": [0.1]
        },
        "select_col_indexes": -1
    }
    hetero_feature_selection_0 = HeteroFeatureSelection(**param)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    # set dataio_1 to replicate model from dataio_0
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data), model=Model(dataio_0.output.model))

    # set train & validate data of hetero_lr_0 component

    pipeline.add_component(secureboost_0, data=Data(train_data=dataio_0.output.data,
                                                    validate_data=dataio_1.output.data))

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=dataio_0.output.data),
                           model=Model(isometric_model=[secureboost_0.output.model]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("homo_feature_selection_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
