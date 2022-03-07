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
from pipeline.component import HomoOneHotEncoder
from pipeline.component.homo_feature_binning import HomoFeatureBinning
from pipeline.component import FederatedSample
from pipeline.component import HomoLR
from pipeline.component import HomoSecureBoost
from pipeline.component import LocalBaseline
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_homo_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_homo_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_1.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0", with_label=True)
    data_transform_1 = DataTransform(name="data_transform_1")

    federated_sample_0 = FederatedSample(name="federated_sample_0", mode="stratified", method="downsample",
                                         fractions=[[0, 1.0], [1, 1.0]], task_type="homo")

    homo_binning_0 = HomoFeatureBinning(name='homo_binning_0', sample_bins=10, method="recursive_query")
    homo_binning_1 = HomoFeatureBinning(name='homo_binning_1')

    homo_onehot_0 = HomoOneHotEncoder(name='homo_onehot_0', need_alignment=True)
    homo_onehot_1 = HomoOneHotEncoder(name='homo_onehot_1')

    homo_lr_0 = HomoLR(name="homo_lr_0", penalty="L2", tol=0.0001, alpha=1.0,
                       optimizer="rmsprop", max_iter=5)
    homo_lr_1 = HomoLR(name="homo_lr_1")

    local_baseline_0 = LocalBaseline(name="local_baseline_0", model_name="LogisticRegression",
                                     model_opts={"penalty": "l2", "tol": 0.0001, "C": 1.0, "fit_intercept": True,
                                                 "solver": "lbfgs", "max_iter": 5, "multi_class": "ovr"})
    local_baseline_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=True)
    local_baseline_0.get_party_instance(role='host', party_id=host).component_param(need_run=True)
    local_baseline_1 = LocalBaseline(name="local_baseline_1")

    homo_secureboost_0 = HomoSecureBoost(name="homo_secureboost_0", num_trees=3)
    homo_secureboost_1 = HomoSecureBoost(name="homo_secureboost_1", num_trees=3)

    evaluation_0 = Evaluation(name="evaluation_0")
    evaluation_1 = Evaluation(name="evaluation_1")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)

    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1, data=Data(data=reader_1.output.data),
                           model=Model(model=data_transform_0.output.model))

    pipeline.add_component(federated_sample_0, data=Data(data=data_transform_0.output.data))

    pipeline.add_component(homo_binning_0, data=Data(data=federated_sample_0.output.data))
    pipeline.add_component(homo_binning_1, data=Data(data=data_transform_1.output.data),
                           model=Model(model=homo_binning_0.output.model))

    pipeline.add_component(homo_onehot_0, data=Data(data=homo_binning_0.output.data))
    pipeline.add_component(homo_onehot_1, data=Data(data=homo_binning_1.output.data),
                           model=Model(model=homo_onehot_0.output.model))

    pipeline.add_component(homo_lr_0, data=Data(data=homo_onehot_0.output.data))
    pipeline.add_component(homo_lr_1, data=Data(data=homo_onehot_1.output.data),
                           model=Model(model=homo_lr_0.output.model))

    pipeline.add_component(local_baseline_0, data=Data(data=homo_onehot_0.output.data))
    pipeline.add_component(local_baseline_1, data=Data(data=homo_onehot_1.output.data),
                           model=Model(model=local_baseline_0.output.model))

    pipeline.add_component(homo_secureboost_0, data=Data(data=homo_onehot_0.output.data))
    pipeline.add_component(homo_secureboost_1, data=Data(data=homo_onehot_1.output.data),
                           model=Model(model=homo_secureboost_0.output.model))

    pipeline.add_component(evaluation_0,
                           data=Data(
                               data=[homo_lr_0.output.data, homo_lr_1.output.data,
                                     local_baseline_0.output.data, local_baseline_1.output.data]))
    pipeline.add_component(evaluation_1,
                           data=Data(
                               data=[homo_secureboost_0.output.data, homo_secureboost_1.output.data]))

    pipeline.compile()

    pipeline.fit()

    print(pipeline.get_component("evaluation_0").get_summary())
    print(pipeline.get_component("evaluation_1").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
