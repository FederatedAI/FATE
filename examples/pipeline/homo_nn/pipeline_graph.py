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

import pathlib
import sys
from torch import log_softmax, logspace, softmax
from torch.nn.modules import Linear, ReLU, Sigmoid, LogSoftmax
from torch.optim import Adam
from pipeline.component.graphNN import GraphNN, GCNLayer
import torch.sparse

additional_path = pathlib.Path(__file__).resolve().parent.parent.resolve().__str__()
if additional_path not in sys.path:
    sys.path.append(additional_path)

dataset =  {
    "guest":{
            "feats": {"name": "cora_feats_guest", "namespace": "experiment"},
            "adj": {"name": "cora_adj_guest", "namespace": "experiment"}
        },
    "host": {
            "feats": {"name": "cora_feats_host", "namespace": "experiment"},
            "adj": {"name": "cora_adj_host", "namespace": "experiment"},
        },
}

from homo_nn._common_component import run_graphnn_pipeline


def main(config="../../config.yaml"):
    gin = GraphNN(name="GraphNN", max_iter=1000, batch_size=-1, early_stop={"early_stop": "diff", "eps": 0.0001})
    gin.add(GCNLayer(in_features=1433, out_features=16, bias=True))
    gin.add(ReLU())
    gin.add(GCNLayer(in_features=16, out_features=7, bias=True))
    gin.add(LogSoftmax())
    gin.compile(
        optimizer='{"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": "False"}', 
        metrics=["Hinge", "accuracy", "AUC"],
        loss="NLLLoss",
    )
    run_graphnn_pipeline(config, dataset, gin)