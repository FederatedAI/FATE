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
from federatedml.secureprotol.spdz.tensor.table_fix_point import TableTensor, table_dot

MODEL_META_NAME = "HeteroPearsonModelMeta"
MODEL_PARAM_NAME = "HeteroPearsonModelParam"


class Pearson(ModelBase):

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

    @staticmethod
    def _standardized(data_instance):
        n = data_instance.count()
        sum_x, sum_square_x = data_instance.mapValues(lambda inst: (inst.features, inst.features ** 2)) \
            .reduce(lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1]))
        mu = sum_x / n
        sigma = np.sqrt(sum_square_x / n - mu ** 2)
        if (sigma <= 0).any():
            raise ValueError(f"zero standard deviation detected, sigma={sigma}")
        return data_instance.mapValues(lambda inst: (inst.features - mu) / sigma)

    def fit(self, data_instance):
        self.names = data_instance.schema["header"]
        data_instance = self._standardized(data_instance)
        self.local_corr = table_dot(data_instance, data_instance)

        with SPDZ("a name") as spdz:
            source = [data_instance, self._other_party]
            if self._local_party.role == "host":
                source = source[::-1]
            x, y = TableTensor.from_source("x", source[0]), TableTensor.from_source("y", source[1])

            m1 = len(x.value.first()[1])
            m2 = len(x.value.first()[1])
            self.shapes.append(m1)
            self.shapes.append(m2)

            n1 = x.value.count()
            n2 = y.value.count()
            if n1 != n2:
                raise ValueError(f"shape miss matched, ({n1, m1}), ({n2, n2})")

            self.corr = x.dot(y, "corr").get() / n1
            self.local_corr /= n1
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
        for party in self._parties:
            param_pb.parties.append(f"({party.role},{party.party_id})")
        param_pb.shape = self.local_corr.shape[0]
        for shape in self.shapes:
            param_pb.shapes.append(shape)
        for name in self.names:
            param_pb.names.append(name)
        for v in self.corr.reshape(-1):
            param_pb.corr.append(max(-1.0, min(float(v), 1.0)))
        for v in self.local_corr.reshape(-1):
            param_pb.local_corr.append(max(-1.0, min(float(v), 1.0)))
        return param_pb

    def export_model(self):
        return self._build_model_dict(meta=self._get_meta(), param=self._get_param())

    def _callback(self):

        self.tracker.set_metric_meta(metric_namespace="statistic",
                                     metric_name="correlation",
                                     metric_meta=MetricMeta(name="pearson",
                                                            metric_type="CORRELATION_GRAPH"))
        self.tracker.log_metric_data(metric_namespace="statistic",
                                     metric_name="correlation",
                                     metrics=self.callback_metrics)
