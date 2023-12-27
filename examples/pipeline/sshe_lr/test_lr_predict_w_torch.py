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

import argparse

import torch

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import Evaluation, Reader
from fate_client.pipeline.components.fate import SSHELR, PSI
from fate_client.pipeline.utils import test_utils


class LogisticRegression(torch.nn.Module):
    def __init__(self, coefficients):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(coefficients.shape[1], 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]

    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host)
    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    reader_0 = Reader("reader_0")
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_0.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )
    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])
    lr_0 = SSHELR("lr_0",
                  epochs=2,
                  learning_rate=0.15,
                  batch_size=300,
                  init_param={"fit_intercept": True, "method": "random_uniform"},
                  train_data=psi_0.outputs["output_data"],
                  reveal_every_epoch=False
                  )

    evaluation_0 = Evaluation("evaluation_0",
                              runtime_parties=dict(guest=guest),
                              default_eval_setting="binary",
                              input_data=lr_0.outputs["train_output_data"])

    pipeline.add_tasks([reader_0, psi_0, lr_0, evaluation_0])

    pipeline.compile()
    pipeline.fit()

    lr_model = pipeline.get_task_info('lr_0').get_output_model()
    param = lr_model['output_model']['data']['estimator']['param']
    dtype = getattr(torch, param['dtype'])
    coef = torch.transpose(torch.tensor(param['coef_'], dtype=dtype), 0, 1)
    intercept = torch.tensor(param["intercept_"], dtype=dtype)

    import pandas as pd

    input_data = pd.read_csv("../../data/breast_hetero_guest.csv", index_col="id")
    input_data.drop(['y'], axis=1, inplace=True)
    input_data = torch.tensor(input_data.values, dtype=dtype)

    pytorch_model = LogisticRegression(coef)
    with torch.no_grad():
        pytorch_model.linear.weight.copy_(coef)
        pytorch_model.linear.bias.copy_(intercept)
        predict_result = pytorch_model(input_data)
    print(f"predictions shape: {predict_result.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
