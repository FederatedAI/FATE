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
import numpy as np

from arch.api import federation
from fate_flow.entity.metric import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.pearson_param import PearsonParam
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor, table_dot

MODEL_META_NAME = "HeteroPearsonModelMeta"
MODEL_PARAM_NAME = "HeteroPearsonModelParam"


class HeteroPearson(ModelBase):

    def __init__(self):
        super().__init__()
        self.model_param = PearsonParam()
        self.role = None
        self.callback_metrics = []
        self.corr = None
        self.local_corr = None
        self._parties = federation.all_parties()
        self._local_party = federation.local_party()
        self._other_party = self._parties[0] if self._parties[0] != self._local_party else self._parties[1]
        self.shapes = []
        self.names = []
        assert len(self._parties) == 2, "support 2 parties only"

    def _init_model(self, param):
        super()._init_model(param)
        self.model_param = param

    def _select_columns(self, data_instance):
        col_names = data_instance.schema["header"]
        if self.model_param.column_indexes == -1:
            self.names = col_names
            name_set = set(self.names)
            for name in self.model_param.column_names:
                if name not in name_set:
                    raise ValueError(f"name={name} not found in header")
            return data_instance.mapValues(lambda inst: inst.features)

        name_to_idx = {col_names[i]: i for i in range(len(col_names))}
        selected = set()
        for name in self.model_param.column_names:
            if name in name_to_idx:
                selected.add(name_to_idx[name])
                continue
            raise ValueError(f"{name} not found")
        for idx in self.model_param.column_indexes:
            if 0 <= idx < len(col_names):
                selected.add(idx)
                continue
            raise ValueError(f"idx={idx} out of bound")
        selected = sorted(list(selected))
        if len(selected) == len(col_names):
            self.names = col_names
            return data_instance.mapValues(lambda inst: inst.features)

        self.names = [col_names[i] for i in selected]
        return data_instance.mapValues(lambda inst: inst.features[selected])

    @staticmethod
    def _standardized(data):
        n = data.count()
        sum_x, sum_square_x = data.mapValues(lambda x: (x, x ** 2)) \
            .reduce(lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1]))
        mu = sum_x / n
        sigma = np.sqrt(sum_square_x / n - mu ** 2)
        if (sigma <= 0).any():
            raise ValueError(f"zero standard deviation detected, sigma={sigma}")
        return n, data.mapValues(lambda x: (x - mu) / sigma)

    def fit(self, data_instance):
        data = self._select_columns(data_instance)
        n, normed = self._standardized(data)
        self.local_corr = table_dot(normed, normed)

        with SPDZ("pearson") as spdz:
            source = [normed, self._other_party]
            if self._local_party.role == "guest":
                x, y = FixedPointTensor.from_source("x", source[0]), FixedPointTensor.from_source("y", source[1])
            else:
                y, x = FixedPointTensor.from_source("y", source[0]), FixedPointTensor.from_source("x", source[1])
            m1 = len(x.value.first()[1])
            m2 = len(y.value.first()[1])
            self.shapes.append(m1)
            self.shapes.append(m2)

            self.corr = spdz.dot(x, y, "corr").get() / n
            self.local_corr /= n
        self._callback()

    @staticmethod
    def _build_model_dict(meta, param):
        return {MODEL_META_NAME: meta, MODEL_PARAM_NAME: param}

    def _get_meta(self):
        from federatedml.protobuf.generated import pearson_model_meta_pb2
        meta_pb = pearson_model_meta_pb2.PearsonModelMeta()
        for shape in self.shapes:
            meta_pb.shapes.append(shape)
        return meta_pb

    def _get_param(self):
        from federatedml.protobuf.generated import pearson_model_param_pb2
        param_pb = pearson_model_param_pb2.PearsonModelParam()
        param_pb.party = f"({self._local_party.role},{self._local_party.party_id})"
        for shape, party in zip(self.shapes, self._parties):
            param_pb.shapes.append(shape)
            param_pb.parties.append(f"({party.role},{party.party_id})")
            _names = param_pb.all_names.add()
            if party == self._local_party:
                for name in self.names:
                    _names.names.append(name)
            else:
                for i in range(shape):
                    _names.names.append(f"{party.role}_{party.party_id}_{i}")
        param_pb.shape = self.local_corr.shape[0]
        for idx, name in enumerate(self.names):
            param_pb.names.append(name)
            anonymous = param_pb.anonymous_map.add()
            anonymous.name = name
            anonymous.anonymous = f"{self._local_party.role}_{self._local_party.party_id}_{idx}"
        for v in self.corr.reshape(-1):
            param_pb.corr.append(max(-1.0, min(float(v), 1.0)))
        for v in self.local_corr.reshape(-1):
            param_pb.local_corr.append(max(-1.0, min(float(v), 1.0)))
        return param_pb

    def export_model(self):
        return self._build_model_dict(meta=self._get_meta(), param=self._get_param())

    # noinspection PyTypeChecker
    def _callback(self):

        self.tracker.set_metric_meta(metric_namespace="statistic",
                                     metric_name="correlation",
                                     metric_meta=MetricMeta(name="pearson",
                                                            metric_type="CORRELATION_GRAPH"))
        self.tracker.log_metric_data(metric_namespace="statistic",
                                     metric_name="correlation",
                                     metrics=self.callback_metrics)
