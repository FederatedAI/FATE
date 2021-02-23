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
from pipeline.component import HeteroLinR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data, Model

from pipeline.utils.tools import load_job_config, JobConfig
from pipeline.runtime.entity import JobParameters

from fate_test.utils import extract_data


def main(config="../../config.yaml", param="./linr_config.yaml", namespace=""):
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

    guest_train_data = {"name": "motor_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "motor_hetero_host", "namespace": f"experiment{namespace}"}

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

    # define DataIO components
    dataio_0 = DataIO(name="dataio_0")  # start component numbering at 0

    # get DataIO party instance of guest
    dataio_0_guest_party_instance = dataio_0.get_party_instance(role='guest', party_id=guest)
    # configure DataIO for guest
    dataio_0_guest_party_instance.component_param(with_label=True, output_format="dense",
                                                  label_name=param["label_name"], label_type="float")
    # get and configure DataIO party instance of host
    dataio_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    # define Intersection component
    intersection_0 = Intersection(name="intersection_0")

    param = {
        "penalty": param["penalty"],
        "validation_freqs": None,
        "early_stopping_rounds": None,
        "max_iter": param["max_iter"],
        "optimizer": param["optimizer"],
        "learning_rate": param["learning_rate"],
        "init_param": param["init_param"],
        "batch_size": param["batch_size"],
        "alpha": param["alpha"]
    }

    hetero_linr_0 = HeteroLinR(name='hetero_linr_0', **param)
    hetero_linr_1 = HeteroLinR(name='hetero_linr_1')

    evaluation_0 = Evaluation(name='evaluation_0', eval_type="regression",
                              metrics=["r2_score",
                                       "mean_squared_error",
                                       "root_mean_squared_error",
                                       "explained_variance"])
    evaluation_1 = Evaluation(name='evaluation_1', eval_type="regression",
                              metrics=["r2_score",
                                       "mean_squared_error",
                                       "root_mean_squared_error",
                                       "explained_variance"])

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_linr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_linr_1,
                           data=Data(test_data=intersection_0.output.data),
                           model=Model(hetero_linr_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_linr_0.output.data))
    pipeline.add_component(evaluation_1, data=Data(data=hetero_linr_1.output.data))


    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)

    metric_summary_linr_0 = pipeline.get_component("evaluation_0").get_summary()
    metric_summary_linr_1 = pipeline.get_component("evaluation_1").get_summary()
    import copy
    metric_linr_0 = copy.copy(metric_summary_linr_0)
    metric_linr_1 = copy.copy(metric_summary_linr_1)

    data_linr_0 = extract_data(pipeline.get_component("hetero_linr_0").get_output_data().get("data"), "predict_result")
    data_linr_1 = extract_data(pipeline.get_component("hetero_linr_1").get_output_data().get("data"), "predict_result")
    import numpy as np
    metric_linr_0["max"] = np.max(data_linr_0)
    metric_linr_0["mean"] = np.mean(data_linr_0)
    metric_linr_0["min"] = np.min(data_linr_0)

    metric_linr_1["max"] = np.max(data_linr_1)
    metric_linr_1["mean"] = np.mean(data_linr_1)
    metric_linr_1["min"] = np.min(data_linr_1)

    metric_summary = copy.copy(metric_summary_linr_0)
    metric_summary["script_metrics"] = {"linr_train": metric_linr_0,
                                        "linr_validate": metric_linr_1}

    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    return data_summary, metric_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY FATE JOB")
    parser.add_argument("-config", type=str,
                        help="config file")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.param)
    else:
        main()
