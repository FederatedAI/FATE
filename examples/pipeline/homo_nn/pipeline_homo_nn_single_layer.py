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
import os
import sys

from pipeline.component.homo_nn import HomoNN
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

additional_path = os.path.realpath('../')
if additional_path not in sys.path:
    sys.path.append(additional_path)

from homo_nn._common_component import run_homo_nn_pipeline, dataset


def main(config="../../config.yaml", namespace=""):
    homo_nn_0 = HomoNN(name="homo_nn_0", max_iter=10, batch_size=-1, early_stop={"early_stop": "diff", "eps": 0.0001})
    homo_nn_0.add(Dense(units=1, input_shape=(10,), activation="sigmoid"))
    homo_nn_0.compile(optimizer=optimizers.Adam(learning_rate=0.05), metrics=["accuracy", "AUC"],
                      loss="binary_crossentropy")
    run_homo_nn_pipeline(config, namespace, dataset.breast, homo_nn_0, 1)
