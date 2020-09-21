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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from string import Template


def build_model():
    model = Sequential()
    model.add(Dense(units=1, input_shape=(30, )))
    return model


def save_runtime_conf(model):
    nn_define = model.to_json()
    temp = open("test_homo_nn_keras_temperate.json").read()
    json = Template(temp).substitute(nn_define=nn_define)

    with open("test_homo_nn_keras.json", "w") as f:
        f.write(json)


if __name__ == '__main__':
    model = build_model()
    save_runtime_conf(model)
