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

import csv

import numpy as np
from sklearn.preprocessing.data import StandardScaler, OneHotEncoder

from federatedml.ftl.data_util.common_data_util import balance_X_y, shuffle_X_y, generate_table_namespace_n_name, \
    create_guest_host_data_generator, split_into_guest_host_dtable


def load_UCI_Credit_Card_data(infile=None, balanced=True, seed=5):
    X = []
    y = []
    sids = []

    with open(infile, "r") as fi:
        fi.readline()
        reader = csv.reader(fi)
        for row in reader:
            sids.append(row[0])
            X.append(row[1:-1])
            y0 = int(row[-1])
            if y0 == 0:
                y0 = -1
            y.append(y0)
    y = np.array(y)

    if balanced:
        X, y = balance_X_y(X, y, seed)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    encoder = OneHotEncoder(categorical_features=[1, 2, 3])
    encoder.fit(X)
    X = encoder.transform(X).toarray()

    X, y = shuffle_X_y(X, y, seed)

    scale_model = StandardScaler()
    X = scale_model.fit_transform(X)

    return X, np.expand_dims(y, axis=1)


def load_guest_host_generators_for_UCI_Credit_Card(file_path, num_samples=None, overlap_ratio=0.2,
                                                   guest_split_ratio=0.5, guest_feature_num=16, balanced=True):
    X, y = load_UCI_Credit_Card_data(infile=file_path, balanced=balanced)

    if num_samples is not None:
        X = X[:num_samples]
        y = y[:num_samples]

    guest_data_generator, host_data_generator, overlap_indexes = create_guest_host_data_generator(X, y,
                                                                                                  overlap_ratio=overlap_ratio,
                                                                                                  guest_split_ratio=guest_split_ratio,
                                                                                                  guest_feature_num=guest_feature_num)

    return guest_data_generator, host_data_generator, overlap_indexes


def load_guest_host_dtable_from_UCI_Credit_Card(data_model_param_dict: dict):
    file_path = data_model_param_dict["file_path"]
    overlap_ratio = data_model_param_dict["overlap_ratio"]
    guest_split_ratio = data_model_param_dict["guest_split_ratio"]
    guest_feature_num = data_model_param_dict["n_feature_guest"]
    num_samples = data_model_param_dict["num_samples"]
    balanced = data_model_param_dict["balanced"]

    namespace, table_name = generate_table_namespace_n_name(file_path)
    tables_name = {
        "guest_table_ns": "guest_" + namespace,
        "guest_table_name": "guest_" + table_name,
        "host_table_ns": "host_" + namespace,
        "host_table_name": "host_" + table_name,
    }

    guest_data, host_data = _load_guest_host_dtable_from_UCI_Credit_Card(file_path=file_path,
                                                                         num_samples=num_samples,
                                                                         tables_name=tables_name,
                                                                         overlap_ratio=overlap_ratio,
                                                                         guest_split_ratio=guest_split_ratio,
                                                                         guest_feature_num=guest_feature_num,
                                                                         balanced=balanced)
    return guest_data, host_data


def _load_guest_host_dtable_from_UCI_Credit_Card(file_path, tables_name, num_samples=None, overlap_ratio=0.2,
                                                 guest_split_ratio=0.5, guest_feature_num=16, balanced=True):
    X, y = load_UCI_Credit_Card_data(infile=file_path, balanced=balanced)

    if num_samples is not None:
        X = X[:num_samples]
        y = y[:num_samples]

    guest_data, host_data, _ = split_into_guest_host_dtable(X, y, overlap_ratio=overlap_ratio,
                                                            guest_split_ratio=guest_split_ratio,
                                                            guest_feature_num=guest_feature_num,
                                                            tables_name=tables_name)

    return guest_data, host_data
