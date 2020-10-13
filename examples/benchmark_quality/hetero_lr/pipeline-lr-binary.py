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
from pipeline.component import DataIO
from pipeline.component import Evaluation
from pipeline.component import HeteroLR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config, JobConfig


def main(config="../../config.yaml", param="./lr_config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    assert isinstance(param, dict)

    data_set = param.get("data_guest").split('/')[-1]
    if data_set == "default_credit_hetero_guest.csv":
        guest_data_table = 'default_credit_hetero_guest'
        host_data_table = 'default_credit_hetero_host'
    elif data_set == 'breast_hetero_guest.csv':
        guest_data_table = 'breast_hetero_guest'
        host_data_table = 'breast_hetero_host'
    elif data_set == 'give_credit_hetero_guest.csv':
        guest_data_table = 'give_credit_hetero_guest'
        host_data_table = 'give_credit_hetero_host'
    elif data_set == 'epsilon_5k_hetero_guest.csv':
        guest_data_table = 'epsilon_5k_hetero_guest'
        host_data_table = 'epsilon_5k_hetero_host'
    else:
        raise ValueError(f"Cannot recognized data_set: {data_set}")

    guest_train_data = {"name": guest_data_table, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_data_table, "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role='guest', party_id=guest)
    # set participants information
    pipeline.set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.algorithm_param(with_label=True, output_format="dense")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    # define Intersection component
    intersection_0 = Intersection(name="intersection_0")

    lr_param = {
        "validation_freqs": None,
        "early_stopping_rounds": None,
    }

    config_param = {
        "penalty": param["penalty"],
        "max_iter": param["max_iter"],
        "alpha": param["alpha"],
        "learning_rate": param["learning_rate"],
        "optimizer": param["optimizer"],
        "batch_size": param["batch_size"],
        "early_stop": "diff",
        "init_param": {
            "init_method": param.get("init_method", 'random_uniform')
        }
    }
    lr_param.update(config_param)
    print(f"lr_param: {lr_param}, data_set: {data_set}")
    hetero_lr_0 = HeteroLR(name='hetero_lr_0', **lr_param)

    evaluation_0 = Evaluation(name='evaluation_0', eval_type="binary")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    print(pipeline.get_component("evaluation_0").get_summary())
    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    result_summary = pipeline.get_component("evaluation_0").get_summary()
    return data_summary, result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)

