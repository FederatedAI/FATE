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
from tensorflow import keras
from string import Template


def build_host_bottom_model():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2, ), activation='relu', kernel_initializer=keras.initializers.Constant(value=1)))
    # model.add(Dense(units=2, input_shape=(20, ), activation='relu',kernel_initializer='random_uniform'))
    return model


def build_guest_bottom_model():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(1, ), activation='relu', kernel_initializer=keras.initializers.Constant(value=1)))
    # model.add(Dense(units=2, input_shape=(10, ), activation='relu', kernel_initializer='random_uniform'))
    return model


def build_top_model():
    model = Sequential()
    model.add(Dense(units=1, input_shape=(2, ), activation='sigmoid', kernel_initializer=keras.initializers.Constant(value=1)))
    # model.add(Dense(units=1, input_shape=(2, ), activation='sigmoid', kernel_initializer='random_uniform'))
    return model


def build_interactive_model():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2, ), activation='relu', kernel_initializer=keras.initializers.Constant(value=1)))
    # model.add(Dense(units=2, input_shape=(2, ), activation='relu', kernel_initializer='random_uniform'))
    return model


def save_runtime_conf():
    guest_bottom_nn_define = build_guest_bottom_model().to_json()
    host_bottom_nn_define = build_host_bottom_model().to_json()
    top_nn_define = build_top_model().to_json()
    interactive_nn_define = build_interactive_model().to_json()

    temp = open("test_hetero_nn_keras_temperate.json").read()
    json = Template(temp).substitute(guest_bottom_nn_define=guest_bottom_nn_define, 
                                     host_bottom_nn_define=host_bottom_nn_define,
                                     top_nn_define=top_nn_define,
                                     interactive_define=interactive_nn_define)

    with open("test_hetero_nn_keras.json", "w") as f:
        f.write(json)


if __name__ == '__main__':
    save_runtime_conf()
