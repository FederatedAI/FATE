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

import torch as t
from torch import nn

from pipeline import fate_torch_hook
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config

fate_torch_hook(t)


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(
        name="intersection_0",
        intersect_method="rsa",
        rsa_params={"hash_method": "sha256", "final_hash_method": "sha256", "key_length": 1024})

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=20,
                           interactive_layer_lr=0.01, batch_size=-1,
                           validation_freqs=1, task_type='classification',
                           encrypt_param={"key_length": 1024}
                           )
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)

    # define model
    guest_bottom = t.nn.Sequential(
        nn.Linear(10, 4),
        nn.ReLU()
    )

    guest_top = t.nn.Sequential(
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    host_bottom = t.nn.Sequential(
        nn.Linear(20, 4),
        nn.ReLU()
    )

    # use interactive layer after fate_torch_hook
    interactive_layer = t.nn.InteractiveLayer(out_dim=4, guest_dim=4, host_dim=4, host_num=1, dropout=0.2)

    guest_nn_0.add_top_model(guest_top)
    guest_nn_0.add_bottom_model(guest_bottom)
    host_nn_0.add_bottom_model(host_bottom)

    optimizer = t.optim.Adam(lr=0.01)  # you can initialize optimizer without parameters after fate_torch_hook
    loss = t.nn.BCELoss()

    hetero_nn_0.set_interactive_layer(interactive_layer)
    hetero_nn_0.compile(optimizer=optimizer, loss=loss)

    evaluation_0 = Evaluation(name='eval_0', eval_type='binary')

    # define components IO
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))
    pipeline.compile()
    pipeline.fit()

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
