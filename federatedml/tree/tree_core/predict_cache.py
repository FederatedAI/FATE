#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


class PredictDataCache(object):
    def __init__(self):
        self._data_map = {}

    def predict_data_at(self, dataset_key, round):
        if dataset_key not in self._data_map:
            return None

        return self._data_map[dataset_key].data_at(round)

    def predict_data_last_round(self, dataset_key):
        if dataset_key not in self._data_map:
            return -1

        return self._data_map[dataset_key].get_last_round()

    @staticmethod
    def get_data_key(data):
        return data.get_name(), data.get_namespace()

    def add_data(self, dataset_key, f):
        if dataset_key not in self._data_map:
            self._data_map[dataset_key] = DataNode()

        self._data_map[dataset_key].add_data(f)


class DataNode(object):
    def __init__(self):
        self._boost_round = None
        self._f = None

    def get_last_round(self):
        return self._boost_round

    def data_at(self, round):
        if round > self._boost_round:
            return None

        return self._f.mapValues(lambda f_list: f_list[round])

    def add_data(self, f):
        if self._boost_round is None:
            self._boost_round = 0
            self._f = f.mapValues(lambda pred: [pred])
        else:
            self._boost_round += 1
            self._f = self._f.join(f, lambda pre_scores, score: pre_scores + [score])

