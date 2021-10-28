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
import json

from pipeline.backend.pipeline import PipeLine
from pipeline.component import HomoLR
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils.tools import load_job_config


def prettify(response, verbose=True):
    if verbose:
        print(json.dumps(response, indent=4, ensure_ascii=False))
        print()
    return response


def main(config="../../config.yaml", namespace=""):
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host[0]
    arbiter = parties.arbiter[0]
    guest_train_data = {"name": "breast_homo_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_homo_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0", output_format='dense', with_label=True)

    pipeline.add_component(reader_0)

    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))

    lr_param = {
        "penalty": "L2",
        "optimizer": "sgd",
        "tol": 1e-05,
        "alpha": 0.01,
        "early_stop": "diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "decay": 1,
        "decay_sqrt": True,
        "init_param": {
            "init_method": "zeros"
        },
        "encrypt_param": {
            "method": None
        },
        "cv_param": {
            "n_splits": 4,
            "shuffle": True,
            "random_seed": 33,
            "need_cv": False
        },
        "callback_param": {
            "callbacks": ["ModelCheckpoint", "EarlyStopping"]
        }
    }

    homo_lr_0 = HomoLR(name="homo_lr_0", max_iter=3, **lr_param)
    homo_lr_1 = HomoLR(name="homo_lr_1", max_iter=30, **lr_param)

    homo_lr_2 = HomoLR(name="homo_lr_2", max_iter=30, **lr_param)

    pipeline.add_component(homo_lr_0, data=Data(train_data=data_transform_0.output.data))
    pipeline.add_component(homo_lr_1, data=Data(train_data=data_transform_0.output.data),
                           model=Model(model=homo_lr_0.output.model))
    pipeline.add_component(homo_lr_2, data=Data(train_data=data_transform_0.output.data))

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    pipeline.add_component(evaluation_0, data=Data(data=[homo_lr_1.output.data,
                                                         homo_lr_2.output.data]))

    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    prettify(pipeline.get_component("evaluation_0").get_summary())
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
