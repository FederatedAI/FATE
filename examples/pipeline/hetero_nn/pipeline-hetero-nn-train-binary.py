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

from pipeline.runtime.entity import JobParameters
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
from pipeline.utils.tools import load_job_config


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
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    dataio_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=100,
                           interactive_layer_lr=0.15, batch_size=-1, early_stop="diff")
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    guest_nn_0.add_bottom_model(Dense(units=3, input_shape=(10,), activation="relu",
                                      kernel_initializer=initializers.Constant(value=1)))
    guest_nn_0.set_interactve_layer(Dense(units=2, input_shape=(2,),
                                          kernel_initializer=initializers.Constant(value=1)))
    guest_nn_0.add_top_model(Dense(units=1, input_shape=(2,), activation="sigmoid",
                                   kernel_initializer=initializers.Constant(value=1)))
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)
    host_nn_0.add_bottom_model(Dense(units=3, input_shape=(20,), activation="relu",
                                     kernel_initializer=initializers.Constant(value=1)))
    host_nn_0.set_interactve_layer(Dense(units=2, input_shape=(2,),
                                         kernel_initializer=initializers.Constant(value=1)))
    hetero_nn_0.compile(optimizer=optimizers.SGD(lr=0.15), metrics=["AUC"], loss="binary_crossentropy")

    hetero_nn_1 = HeteroNN(name="hetero_nn_1")

    evaluation_0 = Evaluation(name="evaluation_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_nn_1, data=Data(test_data=intersection_0.output.data),
                           model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))

    pipeline.compile()

    job_parameters = JobParameters(backend=backend, work_mode=work_mode)
    pipeline.fit(job_parameters)

    print(hetero_nn_0.get_config(roles={"guest": [guest],
                                        "host": [host]}))
    print(pipeline.get_component("hetero_nn_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
