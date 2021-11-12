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
import os
import sys

cur_path = os.path.realpath(__file__)
for i in range(4):
    cur_path = os.path.dirname(cur_path)
print(f'fate_path: {cur_path}')
sys.path.append(cur_path)

from examples.pipeline.hetero_feature_selection import common_tools
from pipeline.utils.tools import load_job_config

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroFeatureSelection
from pipeline.component import HomoLR
from pipeline.component import FeatureScale
from pipeline.component import HomoSecureBoost
from pipeline.component import Evaluation
from pipeline.component import Reader

from pipeline.interface.data import Data
from pipeline.interface.model import Model


def make_normal_dsl(config, namespace):
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host[0]
    arbiter = parties.arbiter[0]
    guest_train_data = {"name": "breast_homo_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_homo_host", "namespace": f"experiment{namespace}"}

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

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=True)

    scale_0 = FeatureScale(name='scale_0')

    homo_sbt_0 = HomoSecureBoost(name="homo_secureboost_0",
                                 num_trees=3,
                                 task_type='classification',
                                 objective_param={"objective": "cross_entropy"},
                                 tree_param={
                                     "max_depth": 3
                                 },
                                 validation_freqs=1
                                 )

    # define Intersection components
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(scale_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(homo_sbt_0, data=Data(train_data=scale_0.output.data))

    selection_param = {
        "name": "hetero_feature_selection_0",
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "homo_sbt_filter"
        ],
        "sbt_param": {
            "metrics": "feature_importance",
            "filter_type": "threshold",
            "take_high": True,
            "threshold": 0.03
        }}
    feature_selection_0 = HeteroFeatureSelection(**selection_param)
    param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 30,
        "early_stop": "diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "decay": 1,
        "decay_sqrt": True,
        "init_param": {
            "init_method": "zeros"
        },
        "encrypt_param": {
            "method": None
        },
        "cv_param": {
            "n_splits": 4,
            "shuffle": True,
            "random_seed": 33,
            "need_cv": False
        }
    }

    homo_lr_0 = HomoLR(name='homo_lr_0', **param)
    pipeline.add_component(feature_selection_0, data=Data(data=scale_0.output.data),
                           model=Model(isometric_model=homo_sbt_0.output.model))
    pipeline.add_component(homo_lr_0, data=Data(train_data=feature_selection_0.output.data))
    evaluation_0 = Evaluation(name='evaluation_0')
    pipeline.add_component(evaluation_0, data=Data(data=homo_lr_0.output.data))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()
    return pipeline


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    pipeline = make_normal_dsl(config, namespace)
    pipeline.fit()
    common_tools.prettify(pipeline.get_component("hetero_feature_selection_0").get_summary())
    common_tools.prettify(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
