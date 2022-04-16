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
from pipeline.utils.tools import load_job_config
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import FeatureScale
from pipeline.component import FederatedSample
from pipeline.component import HeteroFeatureBinning
from pipeline.component import HeteroFeatureSelection
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import OneHotEncoder
from pipeline.component import Reader
from pipeline.interface import Data


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
        missing_fill=True,
        outlier_replace=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False, missing_fill=True,
                                                                                    outlier_replace=True)

    intersection_0 = Intersection(name="intersection_0")
    federated_sample_0 = FederatedSample(name="federated_sample_0", mode="stratified", method="upsample",
                                         fractions=[[0, 1.5], [1, 2.0]])
    feature_scale_0 = FeatureScale(name="feature_scale_0", method="min_max_scale", mode="normal")
    feature_scale_0.get_party_instance(
        role='guest', party_id=guest).component_param(
        feat_upper=[
            1, 2, 1, 1, 0.5, 1, 2, 2, 1, 1])
    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0")
    hetero_feature_selection_0 = HeteroFeatureSelection(name="hetero_feature_selection_0")
    one_hot_0 = OneHotEncoder(name="one_hot_0")
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="rmsprop", tol=1e-5,
                           init_param={"init_method": "random_uniform"},
                           alpha=0.01, max_iter=10, early_stop="diff", batch_size=320, learning_rate=0.15)
    evaluation_0 = Evaluation(name="evaluation_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(federated_sample_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(feature_scale_0, data=Data(data=federated_sample_0.output.data))
    pipeline.add_component(hetero_feature_binning_0, data=Data(data=feature_scale_0.output.data))
    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data))
    pipeline.add_component(one_hot_0, data=Data(data=hetero_feature_selection_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))
    pipeline.compile()

    pipeline.fit()

    print(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
