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
from pipeline.component.nn import TrainerParam
from pipeline.backend.pipeline import PipeLine
from pipeline.component import HomoNN, Evaluation
from pipeline.component.reader import Reader
from pipeline.interface import Data
from pipeline.component.nn import DatasetParam
import os
fate_torch_hook(t)


def main(config="../../config.yaml", namespace=""):
    fate_project_path = os.getenv("FATE_PROJECT_BASE")
    host = 10000
    guest = 9999
    arbiter = 10000
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host,
                                                                                arbiter=arbiter)
    data_0 = {"name": "cora_guest", "namespace": "experiment"}
    data_1 = {"name": "cora_host", "namespace": "experiment"}

    data_path_0 = fate_project_path + '/examples/data/cora4fate/guest'
    data_path_1 = fate_project_path + '/examples/data/cora4fate/host'
    pipeline.bind_table(name=data_0['name'], namespace=data_0['namespace'], path=data_path_0)
    pipeline.bind_table(name=data_1['name'], namespace=data_1['namespace'], path=data_path_1)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=data_0)
    reader_0.get_party_instance(role='host', party_id=host).component_param(table=data_1)
    dataset_param = DatasetParam("graph",
                                 id_col='id',
                                 label_col='y',
                                 feature_dtype='float',
                                 label_dtype='long',
                                 feats_name='feats.csv',
                                 feats_dataset_col='dataset',
                                 feats_dataset_train='train',
                                 feats_dataset_vali='vali',
                                 feats_dataset_test='test',
                                 adj_name='adj.csv',
                                 adj_src_col='node1',
                                 adj_dst_col='node2')

    model = t.nn.Sequential(
        t.nn.CustModel(module_name='graphsage', class_name='Sage', in_channels=1433, hidden_channels=64, class_num=7)
    )
    loss = nn.NLLLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    homo_graph_0 = HomoNN(
        name="homo_graph_0",
        model=model,
        loss=loss,
        optimizer=optimizer,
        dataset=dataset_param,
        trainer=TrainerParam(trainer_name='fedavg_graph_trainer', epochs=10, batch_size=10,
                             validation_freqs=1, num_neighbors=[11, 11], task_type='multi'),
        torch_seed=100
    )

    pipeline.add_component(reader_0)
    pipeline.add_component(homo_graph_0, data=Data(train_data=reader_0.output.data))
    pipeline.add_component(Evaluation(name='eval_0', eval_type='multi'), data=Data(data=homo_graph_0.output.data))

    pipeline.compile()
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
