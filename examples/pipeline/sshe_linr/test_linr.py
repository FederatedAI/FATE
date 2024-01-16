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

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import SSHELinR, PSI, Evaluation, Reader, FeatureScale
from fate_client.pipeline.utils import test_utils


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

    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(namespace=f"experiment{namespace}", name="motor_hetero_guest")
    reader_0.hosts[0].task_parameters(namespace=f"experiment{namespace}", name="motor_hetero_host")
    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])
    scale_0 = FeatureScale('scale_0', train_data=psi_0.outputs['output_data'], method='standard')
    linr_0 = SSHELinR("linr_0",
                      epochs=2,
                      batch_size=100,
                      init_param={"fit_intercept": True},
                      train_data=scale_0.outputs["train_output_data"],
                      reveal_every_epoch=False,
                      early_stop="diff",
                      reveal_loss_freq=1,
                      learning_rate=0.1)
    evaluation_0 = Evaluation("evaluation_0",
                              runtime_parties=dict(guest=guest),
                              default_eval_setting="regression",
                              input_data=linr_0.outputs["train_output_data"])

    pipeline.add_tasks([reader_0, psi_0, scale_0, linr_0, evaluation_0])
    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()

    pipeline.deploy([psi_0, linr_0])

    predict_pipeline = FateFlowPipeline()

    reader_1 = Reader("reader_1", runtime_parties=dict(guest=guest, host=host))
    reader_1.guest.task_parameters(namespace=f"experiment{namespace}", name="motor_hetero_guest")
    reader_1.hosts[0].task_parameters(namespace=f"experiment{namespace}", name="motor_hetero_host")
    deployed_pipeline = pipeline.get_deployed_pipeline()
    deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]
    predict_pipeline.add_tasks([reader_1, deployed_pipeline])
    predict_pipeline.compile()
    # print("\n\n\n")
    # print(predict_pipeline.compile().get_dag())
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
