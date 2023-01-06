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
#pipeline
from pipeline.component.nn import TrainerParam
from pipeline.backend.pipeline import PipeLine
from pipeline.component.data_transform import DataTransform
from pipeline.component import HomoNN, Evaluation
from pipeline.component.reader import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config

import torch_geometric.nn as pyg

fate_torch_hook(t)
# fate_torch_hook(pyg)




def main(config="../../config.yaml", namespace=""):
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
        trainer=TrainerParam(trainer_name='fedavg_graph_trainer', epochs=10, batch_size=10, 
                validation_freqs=1, num_neighbors=[11, 11], task_type='multi'),
        torch_seed=100
    )
    run_homo_graph_pipeline(config, namespace, dataset.cora, homo_graph_0, 1)

def run_homo_graph_pipeline(config, namespace, data: dict, nn_component, num_host=1):
    if isinstance(config, str):
        config = load_job_config(config)

    guest_all_feats = data["guest"]["all_feats"]
    guest_train = data["guest"]["train"]
    guest_val = data["guest"]["val"]
    guest_test = data["guest"]["test"]
    guest_adj = data["guest"]["adj"]

    host_all_feats = data["host"]["all_feats"]
    host_train = data["host"]["train"]
    host_val = data["host"]["val"]
    host_adj = data["host"]["adj"]
    host_test = data["host"]["test"]

    hosts = config.parties.host[:1]
    pipeline = (
        PipeLine()
        .set_initiator(role="guest", party_id=config.parties.guest[0])
        .set_roles(
            guest=config.parties.guest[0], host=hosts, arbiter=config.parties.arbiter
        )
    )

    # all set
    reader_all_feats = Reader(name="reader_all_feats")
    reader_all_feats.get_party_instance(role="guest", party_id=config.parties.guest[0]).component_param(
        table=guest_all_feats
    )

    reader_all_feats.get_party_instance(role="host", party_id=hosts[0]).component_param(
        table=host_all_feats
    )

    data_transform_all_feats = DataTransform(name="data_transform_all_feats", with_label=True)
    data_transform_all_feats.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=True, output_format="dense")
    data_transform_all_feats.get_party_instance(role="host", party_id=hosts).component_param(
        with_label=True
    )    

    #train set
    reader_train = Reader(name="reader_train")
    reader_train.get_party_instance(role="guest", party_id=config.parties.guest[0]).component_param(
        table=guest_train
    )

    reader_train.get_party_instance(role="host", party_id=hosts[0]).component_param(
        table=host_train
    )

    data_transform_train = DataTransform(name="data_transform_train", with_label=True)
    data_transform_train.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=True, output_format="dense")
    data_transform_train.get_party_instance(role="host", party_id=hosts).component_param(
        with_label=True
    )

    #valid set
    reader_val = Reader(name="reader_val")
    reader_val.get_party_instance(role="guest", party_id=config.parties.guest[0]).component_param(
        table=guest_val
    )

    reader_val.get_party_instance(role="host", party_id=hosts[0]).component_param(
        table=host_val
    )

    data_transform_val = DataTransform(name="data_transform_val", with_label=True)
    data_transform_val.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=True, output_format="dense")
    data_transform_val.get_party_instance(role="host", party_id=hosts).component_param(
        with_label=True
    )

    #test set
    reader_test = Reader(name="reader_test")
    reader_test.get_party_instance(role="guest", party_id=config.parties.guest[0]).component_param(
        table=guest_test
    )
    reader_test.get_party_instance(role="host", party_id=hosts[0]).component_param(
        table=host_test
    )
    data_transform_test = DataTransform(name="data_transform_test", with_label=True)
    data_transform_test.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=True, output_format="dense")
    data_transform_test.get_party_instance(role="host", party_id=hosts).component_param(
        with_label=True
    )

    #adjcent table
    reader_adj = Reader(name="reader_adj")
    reader_adj.get_party_instance(role="guest", party_id=config.parties.guest[0]).component_param(
        table=guest_adj
    )
    reader_adj.get_party_instance(role="host", party_id=hosts[0]).component_param(
        table=host_adj
    )
    data_transform_adj = DataTransform(name="data_transform_adj", with_label=False)
    data_transform_adj.get_party_instance(role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=False, output_format="dense")
    data_transform_adj.get_party_instance(role="host", party_id=hosts
    ).component_param(with_label=False, output_format="dense")

    pipeline.add_component(reader_all_feats)
    pipeline.add_component(data_transform_all_feats, data=Data(data=reader_all_feats.output.data))
    pipeline.add_component(reader_train)
    pipeline.add_component(data_transform_train, data=Data(data=reader_train.output.data))
    pipeline.add_component(reader_val)
    pipeline.add_component(data_transform_val, data=Data(data=reader_val.output.data))

    pipeline.add_component(reader_adj)
    pipeline.add_component(data_transform_adj, data=Data(data=reader_adj.output.data))
    pipeline.add_component(nn_component, data=Data(train_data=[data_transform_all_feats.output.data, data_transform_adj.output.data, data_transform_train.output.data],
                                                    validate_data=[data_transform_all_feats.output.data, data_transform_adj.output.data, data_transform_val.output.data]))
    pipeline.add_component(Evaluation(name='eval_0'), data=Data(data=nn_component.output.data))

    pipeline.compile()
    pipeline.fit()
    print(pipeline.get_component("homo_graph_0").get_summary())


class dataset_meta(type):
    @property
    def cora(cls):
        return {
            "guest":{
                    "all_feats": {"name": "cora_feats_guest", "namespace": "experiment"},
                    "train": {"name": "cora_train_guest", "namespace": "experiment"},
                    "val": {"name": "cora_val_guest", "namespace": "experiment"},
                    "test": {"name": "cora_test_guest", "namespace": "experiment"},
                    "adj": {"name": "cora_adj_guest", "namespace": "experiment"}
                },
            "host": {
                    "all_feats": {"name": "cora_feats_host", "namespace": "experiment"},
                    "train": {"name": "cora_train_host", "namespace": "experiment"},
                    "val": {"name": "cora_val_host", "namespace": "experiment"},
                    "test": {"name": "cora_test_host", "namespace": "experiment"},
                    "adj": {"name": "cora_adj_host", "namespace": "experiment"},
                },
        }    


class dataset(metaclass=dataset_meta):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
