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
from pipeline.component import DataStatistics
from pipeline.component import DataTransform
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import PSI
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
    hosts = parties.host

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts)

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
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))

    binning_param = {
        "method": "quantile",
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 10,
        "bin_indexes": -1,
        "bin_names": None,
        "category_indexes": None,
        "category_names": None,
        "adjustment_factor": 0.5,
        "local_only": False,
        "transform_param": {
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "bin_num"
        }
    }
    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0", **binning_param)
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))

    statistic_param = {
        "statistics": ["95%", "coefficient_of_variance", "stddev"],
        "column_indexes": -1,
        "column_names": []
    }
    statistic_0 = DataStatistics(name="statistic_0", **statistic_param)
    pipeline.add_component(statistic_0, data=Data(data=intersection_0.output.data))

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=hosts).component_param(table=host_eval_data)
    data_transform_1 = DataTransform(name="data_transform_1")
    intersection_1 = Intersection(name="intersection_1")
    pipeline.add_component(reader_1)
    pipeline.add_component(
        data_transform_1, data=Data(
            data=reader_1.output.data), model=Model(
            data_transform_0.output.model))
    pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))
    psi_param = {
        "name": "psi_0",
        "max_bin_num": 20
    }
    psi_0 = PSI(**psi_param)
    pipeline.add_component(psi_0, data=Data(train_data=intersection_0.output.data,
                                            validate_data=intersection_1.output.data))

    secureboost_param = {
        "task_type": "classification",
        "learning_rate": 0.1,
        "num_trees": 5,
        "subsample_feature_rate": 1,
        "n_iter_no_change": False,
        "tol": 0.0001,
        "bin_num": 50,
        "objective_param": {
            "objective": "cross_entropy"
        },
        "encrypt_param": {
            "method": "paillier"
        },
        "predict_param": {
            "threshold": 0.5
        }
    }
    secureboost_0 = HeteroSecureBoost(name="secureboost_0", **secureboost_param)
    pipeline.add_component(secureboost_0, data=Data(train_data=intersection_0.output.data))

    selection_param = {
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "iv_filter",
            "statistic_filter",
            "psi_filter",
            "hetero_sbt_filter"
        ],
        "iv_param": {
            "metrics": ["iv", "iv", "iv"],
            "filter_type": ["threshold", "top_k", "top_percentile"],
            "take_high": True,
            "threshold": [0.03, 15, 0.7],
            "host_thresholds": [[0.15], None, None],
            "select_federated": True
        },
        "statistic_param": {
            "metrics": ["skewness", "skewness", "kurtosis", "median"],
            "filter_type": "threshold",
            "take_high": [True, False, False, True],
            "threshold": [-10, 10, 2, -1.5]
        },
        "psi_param": {
            "metrics": "psi",
            "filter_type": "threshold",
            "take_high": False,
            "threshold": -0.1
        },
        "sbt_param": {
            "metrics": "feature_importance",
            "filter_type": "threshold",
            "take_high": True,
            "threshold": 0.03
        }}
    hetero_feature_selection_0 = HeteroFeatureSelection(name="hetero_feature_selection_0", **selection_param)
    pipeline.add_component(hetero_feature_selection_0, data=Data(data=intersection_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model,
                                                        statistic_0.output.model,
                                                        psi_0.output.model,
                                                        secureboost_0.output.model]))
    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
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
