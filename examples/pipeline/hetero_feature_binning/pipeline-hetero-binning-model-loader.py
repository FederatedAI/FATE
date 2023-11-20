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
import copy

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroFeatureBinning
from pipeline.component import Intersection
from pipeline.component import Reader, ModelLoader
from pipeline.interface import Data, Model
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(
        name="intersection_0",
        intersect_method="rsa",
        rsa_params={"hash_method": "sha256", "final_hash_method": "sha256", "key_length": 1024})

    param = {
        "method": "quantile",
        "optimal_binning_param": {
            "metric_method": "gini",
            "min_bin_pct": 0.05,
            "max_bin_pct": 0.8,
            "init_bucket_method": "quantile",
            "init_bin_nums": 100,
            "mixture": True
        },
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 10,
        "bin_indexes": [0, 1, 2, 3, 5],
        "bin_names": None,
        "category_names": None,
        "adjustment_factor": 0.5,
        "local_only": False,
        "encrypt_param": {
            "key_length": 1024
        },
        "transform_param": {
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "bin_num"
        }
    }

    guest_param = copy.deepcopy(param)
    guest_param["category_indexes"] = [0]

    host_param = copy.deepcopy(param)
    host_param["method"] = "optimal"

    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0", **param)
    hetero_feature_binning_0.get_party_instance(role="guest", party_id=guest).component_param(**guest_param)
    hetero_feature_binning_0.get_party_instance(role="host", party_id=host).component_param(**host_param)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=intersection_0.output.data))

    pipeline.compile()

    pipeline.fit()

    loader_pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    model_param = {
        "model_id": pipeline.get_model_info().model_id,
        "model_version": pipeline.get_model_info().model_version,
        "component_name": "hetero_feature_binning_0"
    }
    model_loader_0 = ModelLoader(name="model_loader_0", **model_param)
    hetero_feature_binning_1 = HeteroFeatureBinning(name="hetero_feature_binning_1", **param)
    hetero_feature_binning_1.get_party_instance(role="host", party_id=host).component_param(
        transform_param={"transform_type": "woe"})
    hetero_feature_binning_1.get_party_instance(role="guest", party_id=guest).component_param(
        **guest_param)

    hetero_feature_binning_2 = HeteroFeatureBinning(name="hetero_feature_binning_2",
                                                    transform_param={"transform_type": "bin_num"})

    # add selected components from train pipeline onto predict pipeline
    # specify data source
    loader_pipeline.add_component(model_loader_0)
    loader_pipeline.add_component(reader_0)
    loader_pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    loader_pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    loader_pipeline.add_component(hetero_feature_binning_1,
                                  data=Data(data=intersection_0.output.data),
                                  model=Model(model=model_loader_0.output.model))
    loader_pipeline.add_component(hetero_feature_binning_2,
                                  data=Data(data=intersection_0.output.data),
                                  model=Model(model=hetero_feature_binning_1.output.model))
    loader_pipeline.compile()

    # run predict model
    loader_pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
