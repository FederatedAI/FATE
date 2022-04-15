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
from pipeline.component.evaluation import Evaluation
from pipeline.component.hetero_sshe_lr import HeteroSSHELR
from pipeline.component.intersection import Intersection
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from pipeline.interface.model import Model
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

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    guest_eval_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_eval_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=hosts)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=hosts).component_param(table=host_train_data)
    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).component_param(table=guest_eval_data)
    reader_1.get_party_instance(role='host', party_id=hosts).component_param(table=host_eval_data)

    data_transform_0 = DataTransform(name="data_transform_0", output_format='dense')
    data_transform_1 = DataTransform(name="data_transform_1", output_format='dense')

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True)
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    # define Intersection components
    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)

    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(data_transform_1, data=Data(data=reader_1.output.data),
                           model=Model(data_transform_0.output.model))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=data_transform_1.output.data))

    lr_param = {
        "name": "hetero_sshe_lr_0",
        "penalty": "L2",
        "optimizer": "rmsprop",
        "tol": 0.0001,
        "alpha": 0.01,
        "max_iter": 30,
        "early_stop": "diff",
        "batch_size": -1,
        "callback_param": {
            "callbacks": [
                "EarlyStopping",
                "PerformanceEvaluate"
            ],
            "validation_freqs": 1,
            "early_stopping_rounds": 3
        },
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "zeros"
        },
        "reveal_strategy": "respectively",
        "reveal_every_iter": True
    }

    hetero_sshe_lr_0 = HeteroSSHELR(**lr_param)
    pipeline.add_component(hetero_sshe_lr_0, data=Data(train_data=intersection_0.output.data,
                                                       validate_data=intersection_1.output.data))

    evaluation_data = [hetero_sshe_lr_0.output.data]
    hetero_sshe_lr_1 = HeteroSSHELR(name='hetero_sshe_lr_1')
    pipeline.add_component(hetero_sshe_lr_1, data=Data(test_data=intersection_1.output.data),
                           model=Model(hetero_sshe_lr_0.output.model))
    evaluation_data.append(hetero_sshe_lr_1.output.data)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")
    pipeline.add_component(evaluation_0, data=Data(data=evaluation_data))

    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    prettify(pipeline.get_component("hetero_sshe_lr_0").get_summary())
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
