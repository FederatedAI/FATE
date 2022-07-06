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
from pipeline.component import Evaluation
from pipeline.component import FeatureScale
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import OneHotEncoder
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    feature_scale_0 = FeatureScale(name='feature_scale_0', method="standard_scale",
                                   need_run=True)
    pipeline.add_component(feature_scale_0, data=Data(data=intersection_0.output.data))

    binning_param = {
        "method": "quantile",
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 10,
        "bin_indexes": -1,
        "adjustment_factor": 0.5,
        "local_only": False,
        "need_run": True,
        "transform_param": {
            "transform_cols": -1,
            "transform_type": "bin_num"
        }
    }
    hetero_feature_binning_0 = HeteroFeatureBinning(name='hetero_feature_binning_0',
                                                    **binning_param)
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=feature_scale_0.output.data))

    selection_param = {
        "select_col_indexes": -1,
        "filter_methods": [
            "manually",
            "iv_value_thres",
            "iv_percentile"
        ],
        "manually_param": {
            "filter_out_indexes": None
        },
        "iv_value_param": {
            "value_threshold": 1.0
        },
        "iv_percentile_param": {
            "percentile_threshold": 0.9
        },
        "need_run": True
    }
    hetero_feature_selection_0 = HeteroFeatureSelection(name='hetero_feature_selection_0',
                                                        **selection_param)
    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model]))

    onehot_param = {
        "transform_col_indexes": -1,
        "transform_col_names": None,
        "need_run": True
    }
    one_hot_encoder_0 = OneHotEncoder(name='one_hot_encoder_0', **onehot_param)
    pipeline.add_component(one_hot_encoder_0, data=Data(data=hetero_feature_selection_0.output.data))

    lr_param = {
        "penalty": "L2",
        "optimizer": "rmsprop",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 10,
        "early_stop": "diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "random_uniform"
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        }
    }
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **lr_param)
    pipeline.add_component(hetero_lr_0, data=Data(train_data=one_hot_encoder_0.output.data))

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

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
