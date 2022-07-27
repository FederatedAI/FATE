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

from collections import OrderedDict
from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroNN
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.component import Evaluation
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config
from pipeline.interface import Model

from pipeline import fate_torch_hook
import torch as t
from torch import nn
from torch.nn import init
from torch import optim
from pipeline import fate_torch as ft

# this is important, modify torch modules so that Sequential model be parsed by pipeline
fate_torch_hook(t)


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(role='guest', party_id=guest).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    # define network structure in torch style #
    # define guest model
    Linear = nn.Linear
    ReLU = nn.ReLU
    guest_bottom_a = Linear(10, 8, True)
    seq = nn.Sequential(
        OrderedDict([
            ('layer_0', guest_bottom_a),
            ('relu_0', ReLU())
        ])
    )

    seq2 = nn.Sequential(
        ReLU(),
        Linear(8, 2, True),
        nn.Softmax(dim=1)  # to use coae in binary task, output unit is 2, and need use softmax to compute probability
    )  # so that we can compute loss using fake labels and 2-dim outputs

    # define host model
    seq3 = nn.Sequential(
        Linear(20, 8, True),
        ReLU()
    )

    # interactive layer
    interactive_layer = Linear(16, 8, True)

    # loss fun
    ce_loss_fn = nn.CrossEntropyLoss()

    # optimizer, after fate torch hook optimizer can be created without parameters
    opt: ft.optim.Adam = optim.Adam(lr=0.01)

    hetero_nn_0 = HeteroNN(name="hetero_nn_0", epochs=30, floating_point_precision=None,
                           interactive_layer_lr=0.1, batch_size=-1, early_stop="diff",
                           coae_param={'enable': True, 'epoch': 100})
    guest_nn_0 = hetero_nn_0.get_party_instance(role='guest', party_id=guest)
    guest_nn_0.add_bottom_model(seq)
    guest_nn_0.add_top_model(seq2)

    guest_nn_0.set_interactve_layer(interactive_layer)
    host_nn_0 = hetero_nn_0.get_party_instance(role='host', party_id=host)
    host_nn_0.add_bottom_model(seq3)

    hetero_nn_0.compile(opt, loss=ce_loss_fn)

    hetero_nn_1 = HeteroNN(name="hetero_nn_1")
    evaluation_0 = Evaluation(name="evaluation_0")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(hetero_nn_1, data=Data(test_data=intersection_0.output.data),
                           model=Model(model=hetero_nn_0.output.model))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))
    pipeline.compile()
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
