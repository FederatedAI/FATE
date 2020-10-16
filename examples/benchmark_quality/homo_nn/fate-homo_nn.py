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
from pipeline.component import DataIO, HomoNN, Evaluation
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config, JobConfig
import tensorflow.keras.layers
from tensorflow.keras import optimizers


class dataset(object):
    breast = {
        "guest": {"name": "breast_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "breast_homo_host", "namespace": "experiment"},
            {"name": "breast_homo_host", "namespace": "experiment"}
        ]
    }
    vehicle = {
        "guest": {"name": "vehicle_scale_homo_guest", "namespace": "experiment"},
        "host": [
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"},
            {"name": "vehicle_scale_homo_host", "namespace": "experiment"}
        ]
    }


def main(config="../../config.yaml", param="param_conf.yaml", namespace=""):
    num_host = 1

    if isinstance(config, str):
        config = load_job_config(config)

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    epoch = param["epoch"]
    lr = param["lr"]
    batch_size = param.get("batch_size", -1)
    optimizer_name = param.get("optimizer", "Adam")
    encode_label = param.get("encode_label", True)
    loss = param.get("loss", "categorical_crossentropy")
    metrics = param.get("metrics", ["accuracy"])
    layers = param["layers"]
    data = getattr(dataset, param.get("dataset", "vehicle"))

    guest_train_data = data["guest"]
    host_train_data = data["host"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    hosts = config.parties.host[:num_host]
    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=config.parties.guest[0]) \
        .set_roles(guest=config.parties.guest[0], host=hosts, arbiter=config.parties.arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=config.parties.guest[0]).algorithm_param(table=guest_train_data)
    for i in range(num_host):
        reader_0.get_party_instance(role='host', party_id=hosts[i]) \
            .algorithm_param(table=host_train_data[i])

    dataio_0 = DataIO(name="dataio_0", with_label=True)
    dataio_0.get_party_instance(role='guest', party_id=config.parties.guest[0]) \
        .algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=hosts).algorithm_param(with_label=True)

    homo_nn_0 = HomoNN(name="homo_nn_0", encode_label=encode_label, max_iter=epoch, batch_size=batch_size,
                       early_stop={"early_stop": "diff", "eps": 0.0})
    for layer_config in layers:
        layer = getattr(tensorflow.keras.layers, layer_config["name"])
        layer_params = layer_config["params"]
        homo_nn_0.add(layer(**layer_params))
        homo_nn_0.compile(optimizer=getattr(optimizers, optimizer_name)(learning_rate=lr), metrics=metrics,
                          loss=loss)

    evaluation_0 = Evaluation(name='evaluation_0', eval_type="multi", metrics=["accuracy", "precision", "recall"])

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(homo_nn_0, data=Data(train_data=dataio_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=homo_nn_0.output.data))
    pipeline.compile()
    pipeline.fit(backend=config.backend, work_mode=config.work_mode)
    metric_summary = pipeline.get_component("evaluation_0").get_summary()
    data_summary = dict(
        train={"guest": guest_train_data["name"], **{f"host_{i}": host_train_data[i]["name"] for i in range(num_host)}},
        test={"guest": guest_train_data["name"], **{f"host_{i}": host_train_data[i]["name"] for i in range(num_host)}}
    )
    return data_summary, metric_summary


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
        main(args.param)
