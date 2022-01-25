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
from pipeline.component import HomoLR
from pipeline.component import HomoOneHotEncoder
from pipeline.component import Reader
from pipeline.component import Evaluation
from pipeline.component import FeatureScale
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils.tools import load_job_config
import json


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "heart_nonscaled_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "heart_nonscaled_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "heart_nonscaled_hetero_test", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "heart_nonscaled_hetero_test", "namespace": f"experiment{namespace}"}

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

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=host).component_param(table=host_eval_data)

    # define DataTransform components
    data_transform_0 = DataTransform(
        name="data_transform_0",
        with_label=True,
        output_format="dense",
        label_name='target')  # start component numbering at 0
    data_transform_1 = DataTransform(name="data_transform_1")

    homo_onehot_param = {
        "transform_col_indexes": [1, 2, 5, 6, 8, 10, 11, 12],
        "transform_col_names": [],
        "need_alignment": True
    }

    homo_onehot_0 = HomoOneHotEncoder(name='homo_onehot_0', **homo_onehot_param)
    homo_onehot_1 = HomoOneHotEncoder(name='homo_onehot_1')

    scale_0 = FeatureScale(name='scale_0', method="standard_scale")
    scale_1 = FeatureScale(name='scale_1')

    homo_lr_param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 3,
        "early_stop": "diff",
        "batch_size": 500,
        "learning_rate": 0.15,
        "decay": 1,
        "decay_sqrt": True,
        "init_param": {
            "init_method": "zeros"
        },
        "encrypt_param": {
            "method": "Paillier"
        },
        "cv_param": {
            "n_splits": 4,
            "shuffle": True,
            "random_seed": 33,
            "need_cv": False
        }
    }

    homo_lr_0 = HomoLR(name='homo_lr_0', **homo_lr_param)
    homo_lr_1 = HomoLR(name='homo_lr_1')

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    # set data_transform_1 to replicate model from data_transform_0
    pipeline.add_component(
        data_transform_1, data=Data(
            data=reader_1.output.data), model=Model(
            data_transform_0.output.model))

    pipeline.add_component(homo_onehot_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(homo_onehot_1, data=Data(data=data_transform_1.output.data),
                           model=Model(homo_onehot_0.output.model))
    pipeline.add_component(scale_0, data=Data(data=homo_onehot_0.output.data))
    pipeline.add_component(scale_1, data=Data(data=homo_onehot_1.output.data),
                           model=Model(scale_0.output.model))
    pipeline.add_component(homo_lr_0, data=Data(train_data=scale_0.output.data))
    pipeline.add_component(homo_lr_1, data=Data(test_data=scale_1.output.data),
                           model=Model(homo_lr_0.output.model))
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    evaluation_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)
    pipeline.add_component(evaluation_0, data=Data(data=[homo_lr_0.output.data,
                                                         homo_lr_1.output.data]))
    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    print(json.dumps(pipeline.get_component("homo_lr_0").get_summary(), indent=4, ensure_ascii=False))
    print(json.dumps(pipeline.get_component("evaluation_0").get_summary(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
