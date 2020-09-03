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
from pipeline.component.dataio import DataIO
from pipeline.component.evaluation import Evaluation
from pipeline.component.hetero_lr import HeteroLR
from pipeline.component.hetero_secureboost import HeteroSecureBoost
from pipeline.component.intersection import Intersection
from pipeline.component.one_hot_encoder import OneHotEncoder
from pipeline.component.reader import Reader
from pipeline.component.sampler import FederatedSample
from pipeline.component.scale import FeatureScale
from pipeline.interface.data import Data

from examples.util.config import Config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = Config.load(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "breast_hetero_host", "namespace": f"experiment{namespace}"}

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)
    reader_0 = Reader(name="reader_0")
    # configure Reader for guest
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    # configure Reader for host
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)


    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    intersect_0 = Intersection(name="intersection_0")
    federated_sample_0 = FederatedSample(name="federated_sample_0", mode="stratified",
          method="upsample", fractions=[[0, 1.5], [1, 2.0]])

    feature_scale_0 = FeatureScale(name="feature_scale_0")
    one_hot_0 = OneHotEncoder(name="one_hot_0")

    hetero_lr_0 = HeteroLR(name="hetero_lr_0", early_stop="weight_diff", max_iter=10)
    hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0", num_trees=5,
                                              tree_param={"max_depth":3, "tol":1e-2})
    evaluation_0 = Evaluation(name="evaluation_0")
    evaluation_1 = Evaluation(name="evaluation_1")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=dataio_0.output.data))
    pipeline.add_component(federated_sample_0, data=Data(data=intersect_0.output.data))
    pipeline.add_component(feature_scale_0, data=Data(data=federated_sample_0.output.data))
    pipeline.add_component(one_hot_0, data=Data(train_data=feature_scale_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=one_hot_0.output.data))
    pipeline.add_component(hetero_secureboost_0, data=Data(train_data=feature_scale_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))
    pipeline.add_component(evaluation_1, data=Data(data=hetero_secureboost_0.output.data))

    # pipeline.set_deploy_end_component([dataio_0])
    # pipeline.deploy_component([dataio_0])

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    print (pipeline.get_component("feature_scale_0").get_model_param())
    print (pipeline.get_component("feature_scale_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
