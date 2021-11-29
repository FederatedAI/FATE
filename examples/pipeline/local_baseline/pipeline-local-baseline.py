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
from pipeline.component import Intersection
from pipeline.component import LocalBaseline
from pipeline.component import Reader
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "vehicle_scale_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "vehicle_scale_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")

    data_transform_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        with_label=True,
        output_format="dense",
        label_type="int",
        label_name="y")
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0", intersect_method="rsa", sync_intersect_ids=True,
                                  only_output_key=False)
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="nesterov_momentum_sgd",
                           tol=0.0001, alpha=0.0001, max_iter=30, batch_size=-1,
                           early_stop="diff", learning_rate=0.15, init_param={"init_method": "zeros"})

    local_baseline_0 = LocalBaseline(name="local_baseline_0", model_name="LogisticRegression",
                                     model_opts={"penalty": "l2", "tol": 0.0001, "C": 1.0, "fit_intercept": True,
                                                 "solver": "lbfgs", "max_iter": 5, "multi_class": "ovr"})
    local_baseline_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=True)
    local_baseline_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="multi", pos_label=1)
    evaluation_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=True)
    evaluation_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(local_baseline_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=[hetero_lr_0.output.data, local_baseline_0.output.data]))

    pipeline.compile()

    pipeline.fit()

    # predict
    pipeline.deploy_component([data_transform_0, intersection_0, hetero_lr_0, local_baseline_0])

    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(
        pipeline, data=Data(
            predict_input={
                pipeline.data_transform_0.input.data: reader_0.output.data}))
    predict_pipeline.add_component(
        evaluation_0,
        data=Data(
            data=[
                hetero_lr_0.output.data,
                local_baseline_0.output.data]))
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
