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
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import OneHotEncoder
from pipeline.component import Union
from pipeline.component import LocalBaseline
from pipeline.component import HeteroLinR
from pipeline.component import HeteroPoisson
from pipeline.component import HeteroSSHELR
from pipeline.component import HeteroSSHEPoisson
from pipeline.component import HeteroSSHELinR
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model


"""Note: This script is used for components regression only"""


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

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_1.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    reader_2 = Reader(name="reader_2")
    reader_2.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_2.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        with_label=True,
        missing_fill=True,
        outlier_replace=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False, missing_fill=True,
                                                                                    outlier_replace=True)
    data_transform_1 = DataTransform(name="data_transform_1")
    data_transform_2 = DataTransform(name="data_transform_2")

    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")
    intersection_2 = Intersection(name="intersection_2")

    union_0 = Union(name="union_0")

    federated_sample_0 = FederatedSample(name="federated_sample_0", mode="stratified", method="downsample",
                                         fractions=[[0, 1.0], [1, 1.0]])

    feature_scale_0 = FeatureScale(name="feature_scale_0")
    feature_scale_1 = FeatureScale(name="feature_scale_1")

    hetero_feature_binning_0 = HeteroFeatureBinning(name="hetero_feature_binning_0")
    hetero_feature_binning_1 = HeteroFeatureBinning(name="hetero_feature_binning_1")

    hetero_feature_selection_0 = HeteroFeatureSelection(name="hetero_feature_selection_0")
    hetero_feature_selection_1 = HeteroFeatureSelection(name="hetero_feature_selection_1")

    one_hot_0 = OneHotEncoder(name="one_hot_0")
    one_hot_1 = OneHotEncoder(name="one_hot_1")

    hetero_lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="rmsprop", tol=1e-5,
                           init_param={"init_method": "random_uniform"},
                           alpha=0.01, max_iter=3, early_stop="diff", batch_size=320, learning_rate=0.15)
    hetero_lr_1 = HeteroLR(name="hetero_lr_1")
    hetero_lr_2 = HeteroLR(name="hetero_lr_2", penalty="L2", optimizer="rmsprop", tol=1e-5,
                           init_param={"init_method": "random_uniform"},
                           alpha=0.01, max_iter=3, early_stop="diff", batch_size=320, learning_rate=0.15,
                           cv_param={"n_splits": 5,
                                     "shuffle": True,
                                     "random_seed": 103,
                                     "need_cv": True})

    hetero_sshe_lr_0 = HeteroSSHELR(name="hetero_sshe_lr_0", reveal_every_iter=True, reveal_strategy="respectively",
                                    penalty="L2", optimizer="rmsprop", tol=1e-5, batch_size=320, learning_rate=0.15,
                                    init_param={"init_method": "random_uniform"}, alpha=0.01, max_iter=3)
    hetero_sshe_lr_1 = HeteroSSHELR(name="hetero_sshe_lr_1")

    local_baseline_0 = LocalBaseline(name="local_baseline_0", model_name="LogisticRegression",
                                     model_opts={"penalty": "l2", "tol": 0.0001, "C": 1.0, "fit_intercept": True,
                                                 "solver": "lbfgs", "max_iter": 5, "multi_class": "ovr"})
    local_baseline_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=True)
    local_baseline_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)
    local_baseline_1 = LocalBaseline(name="local_baseline_1")

    hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0", num_trees=3)
    hetero_secureboost_1 = HeteroSecureBoost(name="hetero_secureboost_1")
    hetero_secureboost_2 = HeteroSecureBoost(name="hetero_secureboost_2", num_trees=3,
                                             cv_param={"shuffle": False, "need_cv": True})

    hetero_linr_0 = HeteroLinR(name="hetero_linr_0", penalty="L2", optimizer="sgd", tol=0.001,
                               alpha=0.01, max_iter=3, early_stop="weight_diff", batch_size=-1,
                               learning_rate=0.15, decay=0.0, decay_sqrt=False,
                               init_param={"init_method": "zeros"},
                               floating_point_precision=23)
    hetero_linr_1 = HeteroLinR(name="hetero_linr_1")

    hetero_sshe_linr_0 = HeteroSSHELinR(name="hetero_sshe_linr_0", max_iter=5, early_stop="weight_diff", batch_size=-1)
    hetero_sshe_linr_1 = HeteroSSHELinR(name="hetero_sshe_linr_1")

    hetero_poisson_0 = HeteroPoisson(name="hetero_poisson_0", early_stop="weight_diff", max_iter=10,
                                     alpha=100.0, batch_size=-1, learning_rate=0.01, optimizer="rmsprop",
                                     exposure_colname="exposure", decay_sqrt=False, tol=0.001,
                                     init_param={"init_method": "zeros"}, penalty="L2")
    hetero_poisson_1 = HeteroPoisson(name="hetero_poisson_1")

    hetero_sshe_poisson_0 = HeteroSSHEPoisson(name="hetero_sshe_poisson_0", max_iter=5)
    hetero_sshe_poisson_1 = HeteroSSHEPoisson(name="hetero_sshe_poisson_1")

    evaluation_0 = Evaluation(name="evaluation_0")
    evaluation_1 = Evaluation(name="evaluation_1")
    evaluation_2 = Evaluation(name="evaluation_2")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(reader_2)

    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1, data=Data(data=reader_1.output.data),
                           model=Model(model=data_transform_0.output.model))
    pipeline.add_component(data_transform_2, data=Data(data=reader_2.output.data),
                           model=Model(model=data_transform_0.output.model))

    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))
    pipeline.add_component(intersection_2, data=Data(data=data_transform_2.output.data))

    pipeline.add_component(union_0, data=Data(data=[intersection_0.output.data, intersection_2.output.data]))

    pipeline.add_component(federated_sample_0, data=Data(data=intersection_1.output.data))

    pipeline.add_component(feature_scale_0, data=Data(data=union_0.output.data))
    pipeline.add_component(feature_scale_1, data=Data(data=federated_sample_0.output.data),
                           model=Model(model=feature_scale_0.output.model))

    pipeline.add_component(hetero_feature_binning_0, data=Data(data=feature_scale_0.output.data))
    pipeline.add_component(hetero_feature_binning_1, data=Data(data=feature_scale_1.output.data),
                           model=Model(model=hetero_feature_binning_0.output.model))

    pipeline.add_component(hetero_feature_selection_0, data=Data(data=hetero_feature_binning_0.output.data))
    pipeline.add_component(hetero_feature_selection_1, data=Data(data=hetero_feature_binning_1.output.data),
                           model=Model(model=hetero_feature_selection_0.output.model))

    pipeline.add_component(one_hot_0, data=Data(data=hetero_feature_selection_0.output.data))
    pipeline.add_component(one_hot_1, data=Data(data=hetero_feature_selection_1.output.data),
                           model=Model(model=one_hot_0.output.model))

    pipeline.add_component(hetero_lr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_lr_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_lr_0.output.model))
    pipeline.add_component(hetero_lr_2, data=Data(train_data=one_hot_0.output.data))

    pipeline.add_component(local_baseline_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(local_baseline_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=local_baseline_0.output.model))

    pipeline.add_component(hetero_sshe_lr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_sshe_lr_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_sshe_lr_0.output.model))

    pipeline.add_component(hetero_secureboost_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_secureboost_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_secureboost_0.output.model))
    pipeline.add_component(hetero_secureboost_2, data=Data(train_data=one_hot_0.output.data))

    pipeline.add_component(hetero_linr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_linr_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_linr_0.output.model))

    pipeline.add_component(hetero_sshe_linr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_sshe_linr_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_sshe_linr_0.output.model))

    pipeline.add_component(hetero_poisson_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_poisson_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_poisson_0.output.model))

    pipeline.add_component(evaluation_0, data=Data(data=[hetero_lr_0.output.data, hetero_lr_1.output.data,
                                                         hetero_sshe_lr_0.output.data, hetero_sshe_lr_1.output.data,
                                                         local_baseline_0.output.data, local_baseline_1.output.data]))

    pipeline.add_component(hetero_sshe_poisson_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_sshe_poisson_1, data=Data(test_data=one_hot_1.output.data),
                           model=Model(model=hetero_sshe_poisson_0.output.model))

    pipeline.add_component(evaluation_1,
                           data=Data(
                               data=[hetero_linr_0.output.data, hetero_linr_1.output.data,
                                     hetero_sshe_linr_0.output.data, hetero_linr_1.output.data]))
    pipeline.add_component(evaluation_2,
                           data=Data(
                               data=[hetero_poisson_0.output.data, hetero_poisson_1.output.data,
                                     hetero_sshe_poisson_0.output.data, hetero_sshe_poisson_1.output.data]))

    pipeline.compile()

    pipeline.fit()

    print(pipeline.get_component("evaluation_0").get_summary())
    print(pipeline.get_component("evaluation_1").get_summary())
    print(pipeline.get_component("evaluation_2").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
