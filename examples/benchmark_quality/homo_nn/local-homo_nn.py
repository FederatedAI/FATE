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
import time

import pandas
from pipeline.utils.tools import JobConfig
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def main(param="param_conf.yaml"):
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    model = Sequential([
        Dense(units=5, input_shape=(18,), activation="relu"),
        Dense(units=4, activation="sigmoid")
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=param["lr"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    data_with_label = pandas.concat([
        pandas.read_csv("../../data/vehicle_scale_homo_guest.csv", index_col=0),
        pandas.read_csv("../../data/vehicle_scale_homo_host.csv", index_col=0)
    ]).values
    data = data_with_label[:, 1:]
    labels = to_categorical(data_with_label[:, 0])
    t = time.time()
    model.fit(data, labels, epochs=param["epoch"], batch_size=846)
    evaluate = model.evaluate(data, labels)
    result = {
        "accuracy": evaluate[1]
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.param is not None:
        main(args.param)
    main()
