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
from pipeline.component import HeteroLR
from pipeline.component import SampleWeight
from pipeline.component import Intersection
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import FeatureScale
from pipeline.component import Reader
from pipeline.interface import Data, Model

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

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        with_label=True,
        label_name="y",
        label_type="int",
        output_format="dense")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    binning_param = {
        "name": 'hetero_feature_binning_0',
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

    selection_param = {
        "name": "hetero_feature_selection_0",
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "iv_value_thres"
        ],

        "iv_value_param": {
            "value_threshold": 0.1
        }}
    hetero_feature_binning_0 = HeteroFeatureBinning(**binning_param)

    hetero_feature_selection_0 = HeteroFeatureSelection(**selection_param)

    sample_weight_0 = SampleWeight(name="sample_weight_0")
    sample_weight_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=True,
                                                                                     class_weight={"0": 1, "1": 2})
    sample_weight_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    feature_scale_0 = FeatureScale(name="feature_scale_0", method="standard_scale", need_run=True)

    hetero_lr_0 = HeteroLR(name="hetero_lr_0", optimizer="nesterov_momentum_sgd", tol=0.001,
                           alpha=0.01, max_iter=20, early_stop="weight_diff", batch_size=-1,
                           learning_rate=0.15,
                           init_param={"init_method": "zeros"})

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary", pos_label=1)
    # evaluation_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(sample_weight_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=sample_weight_0.output.data))
    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data),
                           model=Model(isometric_model=[hetero_feature_binning_0.output.model]))
    pipeline.add_component(feature_scale_0, data=Data(hetero_feature_selection_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=feature_scale_0.output.data))
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
