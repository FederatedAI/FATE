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
from federatedml.secureprotol.spdz.fix_point import FixPointEndec
from federatedml.secureprotol.spdz.tensor import SPDZ

Q_FIELD = 2 << 60
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

    def _prepare(self, data_instance):

        # collect tensor
        normalized = list(data_instance.collect())
        normalized.sort()
        normalized = np.array([v for k, v in normalized])

        local_corr = np.einsum("ij,ik->jk", normalized, normalized)

        # source for x and y, assuming data tensor from guest is x, otherwise, y
        tensor_dict = {}
        if self._local_party.role == "guest":
            tensor_dict["x"] = normalized
            tensor_dict["y"] = self._other_party
        elif self._local_party.role == "host":
            tensor_dict["x"] = self._other_party
            tensor_dict["y"] = normalized
        return tensor_dict, local_corr

    def fit(self, data_instance):
        self.names = data_instance.schema["header"]
        data_instance = self._standardized(data_instance)
        tensor_source_dict, local_corr = self._prepare(data_instance)

        endec = FixPointEndec(Q_FIELD)
        for k, v in tensor_source_dict.items():
            if isinstance(v, np.ndarray):
                tensor_source_dict[k] = endec.encode(v)

        with SPDZ("a name") as spdz:
            x = spdz.share("x", tensor_source_dict["x"])
            y = spdz.share("y", tensor_source_dict["y"])
            n1, m1 = x.shape()
            n2, m2 = y.shape()

            if n1 != n2:
                raise ValueError(f"shape miss matched, ({n1, m1}), ({n2, n2})")
            self.shapes.append(m1)
            self.shapes.append(m2)

            c = spdz.tensor_dot(x, y, "ij,ik->jk").rescontruct()
            self.corr = endec.decode(c) / n1
            self.local_corr = local_corr / n1
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
