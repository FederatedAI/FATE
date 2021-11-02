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
import math

import numpy as np

from fate_arch.session import get_parties
from federatedml.model_base import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.pearson_param import PearsonParam
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor.fixedpoint_table import (
    FixedPointTensor,
    table_dot,
)
from federatedml.util import LOGGER
from federatedml.util.anonymous_generator import generate_anonymous

MODEL_META_NAME = "HeteroPearsonModelMeta"
MODEL_PARAM_NAME = "HeteroPearsonModelParam"


class HeteroPearson(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = PearsonParam()
        self.role = None
        self.corr = None
        self.local_corr = None

        self.shapes = []
        self.names = []
        self.parties = []
        self.local_party = None
        self.other_party = None
        self._set_parties()

        self.local_vif = None  # vif from local features

        self._summary = {}

    def _set_parties(self):
        # since multi-host not supported yet, we assume parties are one from guest and one from host
        parties = []
        guest_parties = get_parties().roles_to_parties(["guest"])
        host_parties = get_parties().roles_to_parties(["host"])
        if len(guest_parties) != 1 or len(host_parties) != 1:
            raise ValueError(
                f"one guest and one host required, "
                f"while {len(guest_parties)} guest and {len(host_parties)} host provided"
            )
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = get_parties().local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

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
        sum_x, sum_square_x = data.mapValues(lambda x: (x, x ** 2)).reduce(
            lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1])
        )
        mu = sum_x / n
        sigma = np.sqrt(sum_square_x / n - mu ** 2)
        if (sigma <= 0).any():
            raise ValueError(f"zero standard deviation detected, sigma={sigma}")
        return n, data.mapValues(lambda x: (x - mu) / sigma)

    @staticmethod
    def _vif_from_pearson_matrix(mat: np.ndarray):
        shape = mat.shape
        if shape[0] != shape[1]:
            raise RuntimeError("accept square matrix only")
        dim = shape[0]
        det = np.linalg.det(mat)

        vif = []
        for i in range(dim):
            ax = [j for j in range(dim) if j != i]
            vif.append(np.linalg.det(mat[ax, :][:, ax]) / det)
        return vif

    @staticmethod
    def _generate_determinant_one_matrix(n: int):
        k = math.exp(
            (math.sqrt(2 * math.log(n)) / 2 + math.log(math.factorial(n - 1))) / (2 * n)
        )
        a = np.random.randn(n, n) / k
        d = np.linalg.det(a)
        return a / d ** (1 / n)

    def fit(self, data_instance):
        # local
        data = self._select_columns(data_instance)
        n, normed = self._standardized(data)
        self.local_corr = table_dot(normed, normed)
        self.local_corr /= n
        if self.model_param.calc_local_vif:
            self.local_vif = self._vif_from_pearson_matrix(self.local_corr)
        self._summary["local_corr"] = self.local_corr.tolist()
        self._summary["num_local_features"] = n

        if self.model_param.cross_parties:
            with SPDZ(
                "pearson",
                local_party=self.local_party,
                all_parties=self.parties,
                use_mix_rand=self.model_param.use_mix_rand,
            ) as spdz:
                source = [normed, self.other_party]
                if self.local_party.role == "guest":
                    x, y = (
                        FixedPointTensor.from_source("x", source[0]),
                        FixedPointTensor.from_source("y", source[1]),
                    )
                else:
                    y, x = (
                        FixedPointTensor.from_source("y", source[0]),
                        FixedPointTensor.from_source("x", source[1]),
                    )
                m1 = len(x.value.first()[1])
                m2 = len(y.value.first()[1])
                self.shapes.append(m1)
                self.shapes.append(m2)

                self.corr = spdz.dot(x, y, "corr").get() / n
                self._summary["corr"] = self.corr.tolist()
                self._summary["num_remote_features"] = (
                    m2 if self.local_party.role == "guest" else m1
                )

        else:
            self.shapes.append(self.local_corr.shape[0])
            self.parties = [self.local_party]

        self._callback()
        self.set_summary(self._summary)

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

        # local
        param_pb.party = f"({self.local_party.role},{self.local_party.party_id})"
        param_pb.shape = self.local_corr.shape[0]
        for v in self.local_corr.reshape(-1):
            param_pb.local_corr.append(max(-1.0, min(float(v), 1.0)))
        for idx, name in enumerate(self.names):
            param_pb.names.append(name)
            anonymous = param_pb.anonymous_map.add()
            anonymous.name = name
            anonymous.anonymous = generate_anonymous(
                fid=idx, party_id=self.local_party.party_id, role=self.local_party.role
            )

        if self.model_param.calc_local_vif:
            for vif_value in self.local_vif:
                param_pb.local_vif.append(vif_value)

        # global
        for shape, party in zip(self.shapes, self.parties):
            param_pb.shapes.append(shape)
            param_pb.parties.append(f"({party.role},{party.party_id})")

            _names = param_pb.all_names.add()
            if party == self.local_party:
                for name in self.names:
                    _names.names.append(name)
            else:
                for i in range(shape):
                    _names.names.append(f"{party.role}_{party.party_id}_{i}")

        if self.model_param.cross_parties:
            for v in self.corr.reshape(-1):
                param_pb.corr.append(max(-1.0, min(float(v), 1.0)))

        param_pb.model_name = "HeteroPearson"

        return param_pb

    def export_model(self):
        if self.model_param.need_run:
            return self._build_model_dict(
                meta=self._get_meta(), param=self._get_param()
            )

    # noinspection PyTypeChecker
    def _callback(self):

        self.tracker.set_metric_meta(
            metric_namespace="statistic",
            metric_name="correlation",
            metric_meta=MetricMeta(name="pearson", metric_type="CORRELATION_GRAPH"),
        )
