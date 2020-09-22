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

from pipeline.utils.tools import load_job_config
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
from pipeline.interface import Model


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_1.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True)
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    dataio_1 = DataIO(name="dataio_1")
    dataio_1.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True)
    dataio_1.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")
    intersection_1 = Intersection(name="intersection_1")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=100, validation_freqs=1,
                           interactive_layer_lr=0.15, batch_size=-1, early_stop="diff",
                           early_stopping_rounds=15, use_first_metric_only=True)

    hetero_nn_0.add_bottom_model(Dense(units=3, input_shape=(10,), activation="relu",
                                       kernel_initializer=initializers.Constant(value=1)))
    hetero_nn_0.set_interactve_layer(Dense(units=2, input_shape=(2,),
                                           kernel_initializer=initializers.Constant(value=1)))
    hetero_nn_0.add_top_model(Dense(units=1, input_shape=(2,), activation="sigmoid",
                                    kernel_initializer=initializers.Constant(value=1)))
    hetero_nn_0.compile(optimizer=optimizers.SGD(lr=0.15), metrics=["AUC"], loss="binary_crossentropy")
    hetero_nn_1 = HeteroNN(name="hetero_nn_1")

    evaluation_0 = Evaluation(name="evaluation_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(intersection_1, data=Data(data=dataio_1.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data,
                                                  validate_data=intersection_1.output.data))
    pipeline.add_component(hetero_nn_1, data=Data(test_data=intersection_1.output.data),
                           model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=[hetero_nn_0.output.data, hetero_nn_1.output.data]))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    print(pipeline.get_component("hetero_nn_0").get_summary())
    print(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
