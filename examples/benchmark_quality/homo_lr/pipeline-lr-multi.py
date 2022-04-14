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
from pipeline.component import HomoLR
from pipeline.component import Reader
from pipeline.interface import Data, Model
from pipeline.utils.tools import load_job_config, JobConfig

from fate_test.utils import extract_data, parse_summary_result
from federatedml.evaluation.metrics import classification_metric


def main(config="../../config.yaml", param="./breast_lr_config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    assert isinstance(param, dict)
    data_set = param.get("data_guest").split('/')[-1]
    if data_set == "vehicle_scale_homo_guest.csv":
        guest_data_table = 'vehicle_scale_homo_guest'
        host_data_table = 'vehicle_scale_homo_host'
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
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    # define DataTransform components
    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0

    # get DataTransform party instance of guest
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role='guest', party_id=guest)
    # configure DataTransform for guest
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    # get and configure DataTransform party instance of host
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=True)

    lr_param = {
    }

    config_param = {
        "penalty": param["penalty"],
        "max_iter": param["max_iter"],
        "alpha": param["alpha"],
        "learning_rate": param["learning_rate"],
        "optimizer": param.get("optimizer", "sgd"),
        "batch_size": param.get("batch_size", -1),
        "init_param": {
            "init_method": param.get("init_method", 'random_uniform')
        },
        "encrypt_param": {
            "method": None
        }
    }
    lr_param.update(config_param)
    print(f"lr_param: {lr_param}, data_set: {data_set}")
    homo_lr_0 = HomoLR(name='homo_lr_0', **lr_param)
    homo_lr_1 = HomoLR(name='homo_lr_1')

    evaluation_0 = Evaluation(name='evaluation_0', eval_type="multi")
    evaluation_0.get_party_instance(role='host', party_id=host).component_param(need_run=False)

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(homo_lr_0, data=Data(train_data=data_transform_0.output.data))
    pipeline.add_component(homo_lr_1, data=Data(test_data=data_transform_0.output.data),
                           model=Model(homo_lr_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=homo_lr_0.output.data))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    pipeline.fit()
    # query component summary
    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    result_summary = parse_summary_result(pipeline.get_component("evaluation_0").get_summary())
    lr_0_data = pipeline.get_component("homo_lr_0").get_output_data().get("data")
    lr_1_data = pipeline.get_component("homo_lr_1").get_output_data().get("data")
    lr_0_score_label = extract_data(lr_0_data, "predict_result", keep_id=True)
    lr_1_score_label = extract_data(lr_1_data, "predict_result", keep_id=True)

    metric_lr = {
        "score_diversity_ratio":
            classification_metric.Distribution.compute(lr_0_score_label, lr_1_score_label)}
    result_summary["distribution_metrics"] = {"homo_lr": metric_lr}

    print(result_summary)

    return data_summary, result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-config", type=str,
                        help="config file")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.param)
    else:
        main()
