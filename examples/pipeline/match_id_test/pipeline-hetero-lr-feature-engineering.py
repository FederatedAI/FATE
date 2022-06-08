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
from pipeline.component import Evaluation, DataStatistics, HeteroPearson
from pipeline.component import HeteroLR, OneHotEncoder
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import FeatureScale
from pipeline.component import Intersection
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
    feature_scale_0 = FeatureScale(name='feature_scale_0', method="standard_scale",
                                   need_run=True)

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

    statistic_0 = DataStatistics(name='statistic_0', statistics=["95%"])
    pearson_0 = HeteroPearson(name='pearson_0', column_indexes=-1)
    onehot_0 = OneHotEncoder(name='onehot_0')
    selection_param = {
        "name": "hetero_feature_selection_0",
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "manually",
            "unique_value",
            "iv_filter",
            "coefficient_of_variation_value_thres",
            "outlier_cols"
        ],
        "manually_param": {
            "filter_out_indexes": [
                0,
                1,
                2
            ],
            "filter_out_names": [
                "x3"
            ]
        },
        "unique_param": {
            "eps": 1e-06
        },
        "iv_param": {
            "metrics": ["iv", "iv", "iv"],
            "filter_type": ["threshold", "top_k", "top_percentile"],
            "threshold": [0.001, 100, 0.99]
        },
        "variance_coe_param": {
            "value_threshold": 0.3
        },
        "outlier_param": {
            "percentile": 0.95,
            "upper_threshold": 2.0
        }}
    hetero_feature_selection_0 = HeteroFeatureSelection(**selection_param)

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
    evaluation_0 = Evaluation(name='evaluation_0')

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(feature_scale_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=feature_scale_0.output.data))
    pipeline.add_component(statistic_0, data=Data(data=feature_scale_0.output.data))
    pipeline.add_component(pearson_0, data=Data(data=feature_scale_0.output.data))

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model,
                                                        statistic_0.output.model]))
    pipeline.add_component(onehot_0, data=Data(data=hetero_feature_selection_0.output.data))

    pipeline.add_component(hetero_lr_0, data=Data(train_data=onehot_0.output.data))
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
