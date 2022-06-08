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
from pipeline.component import DataTransform
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import Evaluation
from pipeline.component import SampleWeight
from pipeline.component import Reader
from pipeline.component import FeatureScale
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

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

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

    # define DataTransform components
    data_transform_0 = DataTransform(
        name="data_transform_0",
        with_label=True,
        output_format="dense")  # start component numbering at 0
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)
    intersect_0 = Intersection(name='intersect_0')

    scale_0 = FeatureScale(name='scale_0', need_run=False)
    sample_weight_0 = SampleWeight(name="sample_weight_0", class_weight={"0": 1, "1": 2})
    sample_weight_0.get_party_instance(role="host", party_id=host).component_param(need_run=False)

    param = {
        "penalty": None,
        "optimizer": "sgd",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 3,
        "early_stop": "diff",
        "batch_size": 320,
        "learning_rate": 0.15,
        "decay": 0,
        "decay_sqrt": True,
        "init_param": {
            "init_method": "ones"
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": True,
            "random_seed": 33,
            "need_cv": False
        }
    }
    hetero_lr_0 = HeteroLR(name='hetero_lr_0', **param)
    evaluation_0 = Evaluation(name='evaluation_0')
    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    # set data input sources of intersection components
    pipeline.add_component(scale_0, data=Data(data=intersect_0.output.data))
    pipeline.add_component(sample_weight_0, data=Data(data=scale_0.output.data))

    pipeline.add_component(hetero_lr_0, data=Data(train_data=sample_weight_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
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
