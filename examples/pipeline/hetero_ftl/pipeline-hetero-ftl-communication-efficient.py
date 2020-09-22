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
from pipeline.component.hetero_ftl import HeteroFTL
from pipeline.component.reader import Reader
from pipeline.interface.data import Data
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from pipeline.component.evaluation import Evaluation

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "nus_wide_guest", "namespace": f"experiment{namespace}"}
    host_train_data = {"name": "nus_wide_host", "namespace": f"experiment{namespace}"}
    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0")
    dataio_0.get_party_instance(role='guest', party_id=guest).algorithm_param(with_label=True, output_format="dense")
    dataio_0.get_party_instance(role='host', party_id=host).algorithm_param(with_label=False)

    hetero_ftl_0 = HeteroFTL(name='hetero_ftl_0',
                             epochs=10,
                             alpha=1,
                             batch_size=-1,
                             mode='plain',
                             communication_efficient=True,
                             local_round=5)

    hetero_ftl_0.add_nn_layer(Dense(units=32, activation='sigmoid',
                                    kernel_initializer=initializers.RandomNormal(stddev=1.0,
                                                                                 dtype="float32"),
                                    bias_initializer=initializers.Zeros()))
    hetero_ftl_0.compile(optimizer=optimizers.Adam(lr=0.01))
    evaluation_0 = Evaluation(name='evaluation_0', eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(hetero_ftl_0, data=Data(train_data=dataio_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_ftl_0.output.data))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)

    """
    # predict

    pipeline.predict(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE,
                     feed_dict={input_0:
                                    {"guest":
                                         {9999: guest_train_data},
                                     "host": {
                                         10000: host_train_data[0]
                                     }
                                     }
                                })
            

    with open("output.pkl", "wb") as fout:
        fout.write(pipeline.dump())
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()