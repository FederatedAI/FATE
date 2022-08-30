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
from pipeline.component.data_transform import DataTransform
from pipeline.component.homo_nn import HomoNN
from pipeline.component.reader import Reader
from pipeline.interface import Data
from pipeline.utils.tools import load_job_config
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense


def main(config="../../config.yaml", namespace=""):
    homo_nn_0 = HomoNN(
        name="homo_nn_0",
        max_iter=10,
        batch_size=-1,
        early_stop={"early_stop": "diff", "eps": 0.0001},
    )
    homo_nn_0.add(Dense(units=6, input_shape=(10,), activation="relu"))
    homo_nn_0.add(Dense(units=1, activation="sigmoid"))
    homo_nn_0.compile(
        optimizer=optimizers.Adam(learning_rate=0.05),
        metrics=["Hinge", "accuracy", "AUC"],
        loss="binary_crossentropy",
    )
    run_homo_nn_pipeline(config, namespace, dataset.breast, homo_nn_0, 1)


def run_homo_nn_pipeline(config, namespace, data: dict, nn_component, num_host):
    if isinstance(config, str):
        config = load_job_config(config)

    guest_train_data = data["guest"]
    host_train_data = data["host"][:num_host]
    for d in [guest_train_data, *host_train_data]:
        d["namespace"] = f"{d['namespace']}{namespace}"

    hosts = config.parties.host[:num_host]
    pipeline = (
        PipeLine()
        .set_initiator(role="guest", party_id=config.parties.guest[0])
        .set_roles(
            guest=config.parties.guest[0], host=hosts, arbiter=config.parties.arbiter
        )
    )

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(table=guest_train_data)
    for i in range(num_host):
        reader_0.get_party_instance(role="host", party_id=hosts[i]).component_param(
            table=host_train_data[i]
        )

    data_transform_0 = DataTransform(name="data_transform_0", with_label=True)
    data_transform_0.get_party_instance(
        role="guest", party_id=config.parties.guest[0]
    ).component_param(with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=hosts).component_param(
        with_label=True
    )

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(
        nn_component, data=Data(train_data=data_transform_0.output.data)
    )
    pipeline.compile()
    pipeline.fit()
    print(pipeline.get_component("homo_nn_0").get_summary())
    pipeline.deploy_component([data_transform_0, nn_component])

    # predict
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_0)
    predict_pipeline.add_component(
        pipeline,
        data=Data(
            predict_input={pipeline.data_transform_0.input.data: reader_0.output.data}
        ),
    )
    # run predict model
    predict_pipeline.predict()


# noinspection PyPep8Naming
class dataset_meta(type):
    @property
    def breast(cls):
        return {
            "guest": {"name": "breast_homo_guest", "namespace": "experiment"},
            "host": [
                {"name": "breast_homo_host", "namespace": "experiment"},
                {"name": "breast_homo_host", "namespace": "experiment"},
            ],
        }

    @property
    def vehicle(cls):
        return {
            "guest": {"name": "vehicle_scale_homo_guest", "namespace": "experiment", },
            "host": [
                {"name": "vehicle_scale_homo_host", "namespace": "experiment"},
                {"name": "vehicle_scale_homo_host", "namespace": "experiment"},
            ],
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
