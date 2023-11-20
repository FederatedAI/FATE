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

# torch
import torch as t
from torch import nn

from pipeline import fate_torch_hook
# pipeline
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, HomoNN, Evaluation
from pipeline.component.nn import TrainerParam
from pipeline.interface import Data, Model
from pipeline.utils.tools import load_job_config

fate_torch_hook(t)


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    train_data_0 = {"name": "breast_homo_guest", "namespace": "experiment"}
    train_data_1 = {"name": "breast_homo_host", "namespace": "experiment"}
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=train_data_0)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=train_data_1)

    data_transform_0 = DataTransform(name='data_transform_0')
    data_transform_0.get_party_instance(
        role='guest', party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(
        role='host', party_id=host).component_param(
        with_label=True, output_format="dense")

    model = nn.Sequential(
        nn.Linear(30, 1),
        nn.Sigmoid()
    )
    loss = nn.BCELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)

    nn_component = HomoNN(name='nn_0',
                          model=model,
                          loss=loss,
                          optimizer=optimizer,
                          trainer=TrainerParam(trainer_name='fedavg_trainer', epochs=20, batch_size=128,
                                               validation_freqs=1),
                          torch_seed=100
                          )

    nn_predict = HomoNN(name='nn_1')

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(nn_component, data=Data(train_data=data_transform_0.output.data))
    pipeline.add_component(nn_predict, data=Data(test_data=data_transform_0.output.data),
                           model=Model(model=nn_component.output.model))
    pipeline.add_component(Evaluation(name='eval_0'), data=Data(data=nn_component.output.data))

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
