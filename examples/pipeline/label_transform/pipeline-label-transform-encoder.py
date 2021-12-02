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

from pipeline.backend.pipeline import PipeLine
from pipeline.component import LabelTransform
from pipeline.component import HeteroLR
from pipeline.component import DataTransform
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data, Model

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    # initialize pipeline
    pipeline = PipeLine()
    # set job initiator
    pipeline.set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    # define Reader components to read in data
    reader_0 = Reader(name="reader_0")
    # configure Reader
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")  # start component numbering at 0
    data_transform_0_guest_party_instance = data_transform_0.get_party_instance(role="guest", party_id=guest)
    data_transform_0_guest_party_instance.component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False,
                                                                                    output_format="dense")
    intersection_0 = Intersection(name="intersection_0")

    label_transform_0 = LabelTransform(name="label_transform_0", label_encoder={"0": 1, "1": 0}, label_list=[0, 1])
    label_transform_0.get_party_instance(role="host", party_id=host).component_param(need_run=False)

    hetero_lr_0 = HeteroLR(name="hetero_lr_0", penalty="L2", optimizer="sgd", tol=0.001,
                           alpha=0.01, max_iter=20, early_stop="weight_diff", batch_size=-1,
                           learning_rate=0.15, decay=0.0, decay_sqrt=False,
                           init_param={"init_method": "zeros"},
                           floating_point_precision=23)

    label_transform_1 = LabelTransform(name="label_transform_1")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(label_transform_0, data=Data(data=intersection_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=label_transform_0.output.data))
    pipeline.add_component(
        label_transform_1, data=Data(
            data=hetero_lr_0.output.data), model=Model(
            label_transform_0.output.model))

    # compile pipeline once finished adding modules, this step will form conf and dsl files for running job
    pipeline.compile()

    # fit model
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
