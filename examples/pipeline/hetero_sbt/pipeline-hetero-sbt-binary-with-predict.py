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
from pipeline.component import HeteroSecureBoost
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.component import Evaluation
from pipeline.interface import Model

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    # data sets
    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_validate_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_validate_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host,)

    # set data reader and data-io

    reader_0, reader_1 = Reader(name="reader_0"), Reader(name="reader_1")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)
    reader_1.get_party_instance(role="guest", party_id=guest).component_param(table=guest_validate_data)
    reader_1.get_party_instance(role="host", party_id=host).component_param(table=host_validate_data)

    data_transform_0, data_transform_1 = DataTransform(name="data_transform_0"), DataTransform(name="data_transform_1")

    data_transform_0.get_party_instance(
        role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)
    data_transform_1.get_party_instance(
        role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_1.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    # data intersect component
    intersect_0 = Intersection(name="intersection_0")
    intersect_1 = Intersection(name="intersection_1")

    # secure boost component
    hetero_secure_boost_0 = HeteroSecureBoost(name="hetero_secure_boost_0",
                                              num_trees=3,
                                              task_type="classification",
                                              objective_param={"objective": "cross_entropy"},
                                              encrypt_param={"method": "Paillier"},
                                              tree_param={"max_depth": 3},
                                              validation_freqs=1)

    # evaluation component
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(
        data_transform_1, data=Data(
            data=reader_1.output.data), model=Model(
            data_transform_0.output.model))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersect_1, data=Data(data=data_transform_1.output.data))
    pipeline.add_component(hetero_secure_boost_0, data=Data(train_data=intersect_0.output.data,
                                                            validate_data=intersect_1.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    pipeline.fit()

    print("fitting hetero secureboost done, result:")
    print(pipeline.get_component("hetero_secure_boost_0").get_summary())

    print('start to predict')

    # predict
    # deploy required components
    pipeline.deploy_component([data_transform_0, intersect_0, hetero_secure_boost_0, evaluation_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_0)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(
        pipeline, data=Data(
            predict_input={
                pipeline.data_transform_0.input.data: reader_0.output.data}))

    # run predict model
    predict_pipeline.predict()
    predict_result = predict_pipeline.get_component("hetero_secure_boost_0").get_output_data()
    print("Showing 10 data of predict result")
    for ret in predict_result["data"][:10]:
        print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
