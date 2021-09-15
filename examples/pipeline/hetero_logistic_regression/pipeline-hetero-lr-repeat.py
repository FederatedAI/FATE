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
from pipeline.component import DataIO
from pipeline.component import Evaluation
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.component import SampleWeight
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "hetero_unnitest_repeated_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "hetero_unnitest_repeated_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True, label_name="y",
                                                                              label_type="int", output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")
    sample_weight_0 = SampleWeight(name="sample_weight_0")
    sample_weight_0.get_party_instance(role='guest', party_id=guest).component_param(need_run=False,
                                                                                     class_weight={"0": 1, "1": 2})
    sample_weight_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)
    param = {
        "penalty": None,
        "optimizer": "sgd",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 3,
        "early_stop": "diff",
        "batch_size": 320,
        "learning_rate": 15,
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

    hetero_lr_0 = HeteroLR(**param)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary", pos_label=1)
    # evaluation_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(sample_weight_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=sample_weight_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    pipeline.compile()

    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)
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
