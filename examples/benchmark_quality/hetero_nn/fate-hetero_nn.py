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

from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataIO
from pipeline.component import Evaluation
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config, JobConfig


def main(config="../../config.yaml", param="./hetero_nn_breast_config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)

    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": param["guest_table_name"], "namespace": f"experiment{namespace}"}
    host_train_data = {"name": param["host_table_name"], "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True)
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=param["epochs"],
                           interactive_layer_lr=param["learning_rate"], batch_size=param["batch_size"], early_stop="diff")
    hetero_nn_0.add_bottom_model(Dense(units=param["bottom_layer_units"], input_shape=(10,), activation="relu",
                                       kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=123)))
    hetero_nn_0.set_interactve_layer(
        Dense(units=param["interactive_layer_units"], input_shape=(param["bottom_layer_units"],),
              kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=123)))
    hetero_nn_0.add_top_model(
        Dense(units=param["top_layer_units"], input_shape=(param["interactive_layer_units"],), activation=param["top_act"],
              kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=123)))
    hetero_nn_0.compile(optimizer=optimizers.SGD(lr=param["learning_rate"]), metrics=param["metrics"],
                        loss=param["loss"])

    if param["loss"] == "categorical_crossentropy":
        eval_type = "multi"
    else:
        eval_type = "binary"

    evaluation_0 = Evaluation(name="evaluation_0", eval_type=eval_type)

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }
    return data_summary, pipeline.get_component("evaluation_0").get_summary()


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
