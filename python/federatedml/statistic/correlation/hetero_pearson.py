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
from fate_arch.common import Party
from federatedml.model_base import MetricMeta, ModelBase
from federatedml.param.pearson_param import PearsonParam
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor.fixedpoint_table import (
    FixedPointTensor,
    table_dot,
)
from federatedml.statistic.data_overview import get_anonymous_header, get_header
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables
from federatedml.util import LOGGER


class PearsonTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.anonymous_host = self._create_variable(
            "anonymous_host", src=["host"], dst=["guest"]
        )
        self.anonymous_guest = self._create_variable(
            "anonymous_guest", src=["guest"], dst=["host"]
        )


class HeteroPearson(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = PearsonParam()
        self.transfer_variable = PearsonTransferVariable()

        self._summary = {}
        self._modelsaver = PearsonModelSaver()

    def fit(self, data_instance):
        LOGGER.info("fit start")
        column_names = get_header(data_instance)
        column_anonymous_names = get_anonymous_header(data_instance)
        self._modelsaver.save_local_anonymous(column_names, column_anonymous_names)
        parties = [
            Party("guest", self.component_properties.guest_partyid),
            Party("host", self.component_properties.host_party_idlist[0]),
        ]
        local_party = parties[0] if self.is_guest else parties[1]
        other_party = parties[1] if self.is_guest else parties[0]
        self._modelsaver.save_party(local_party)

        LOGGER.info("select features")
        names, selected_features = select_columns(
            data_instance,
            self.model_param.column_indexes,
            self.model_param.column_names,
        )

        LOGGER.info("standardized feature data")
        num_data, standardized, remainds_indexes, num_features = standardize(
            selected_features
        )
        self._summary["num_local_features"] = num_features

        # local corr
        LOGGER.info("calculate correlation cross local features")
        local_corr = table_dot(standardized, standardized) / num_data
        fixed_local_corr = fix_local_corr(local_corr, remainds_indexes, num_features)
        self._modelsaver.save_local_corr(fixed_local_corr)
        self._summary["local_corr"] = fixed_local_corr.tolist()
        shape = fixed_local_corr.shape[0]

        # local vif
        if self.model_param.calc_local_vif:
            LOGGER.info("calc_local_vif enabled, calculate vif for local features")
            local_vif = vif_from_pearson_matrix(local_corr)
            fixed_local_vif = fix_vif(local_vif, remainds_indexes, num_features)
            self._modelsaver.save_local_vif(fixed_local_vif)
        else:
            LOGGER.info("calc_local_vif disabled, skip local vif")

        # not cross parties
        if not self.model_param.cross_parties:
            LOGGER.info("cross_parties disabled, save model")
            self._modelsaver.save_party_info(shape, local_party, names)
        # cross parties
        else:
            LOGGER.info(
                "cross_parties enabled, calculating correlation with remote features"
            )
            # sync anonymous
            LOGGER.info("sync anonymous names")
            remote_anonymous_names, remote_remainds_indexes = self.sync_anonymous_names(
                column_anonymous_names, remainds_indexes
            )
            if self.is_guest:
                names = [column_names, remote_anonymous_names]
                remainds_indexes_tuple = (remainds_indexes, remote_remainds_indexes)
            else:
                names = [remote_anonymous_names, column_names]
                remainds_indexes_tuple = (remote_remainds_indexes, remainds_indexes)
            m1, m2 = len(names[0]), len(names[1])
            shapes = [m1, m2]
            for shape, party, name in zip(shapes, parties, names):
                self._modelsaver.save_party_info(shape, party, name)
            self._summary["num_remote_features"] = m2 if self.is_guest else m1

            with SPDZ(
                "pearson",
                local_party=local_party,
                all_parties=parties,
                use_mix_rand=self.model_param.use_mix_rand,
            ) as spdz:
                LOGGER.info("secret share: prepare data")
                if self.is_guest:
                    x, y = (
                        FixedPointTensor.from_source("x", standardized),
                        FixedPointTensor.from_source("y", other_party),
                    )
                else:
                    y, x = (
                        FixedPointTensor.from_source("y", standardized),
                        FixedPointTensor.from_source("x", other_party),
                    )
                LOGGER.info("secret share: dot")
                corr = spdz.dot(x, y, "corr").get() / num_data
                fixed_corr = fix_corr(
                    corr, m1, m2, remainds_indexes_tuple[0], remainds_indexes_tuple[1]
                )
                self._modelsaver.save_cross_corr(fixed_corr)
                self._summary["corr"] = fixed_corr.tolist()

        self._callback()
        self.set_summary(self._summary)
        LOGGER.info("fit done")

    @property
    def is_guest(self):
        return self.component_properties.role == "guest"

    def _init_model(self, param):
        super()._init_model(param)
        self.model_param = param

    def export_model(self):
        return self._modelsaver.export()

    # noinspection PyTypeChecker

    def _callback(self):

        self.tracker.set_metric_meta(
            metric_namespace="statistic",
            metric_name="correlation",
            metric_meta=MetricMeta(name="pearson", metric_type="CORRELATION_GRAPH"),
        )

    def sync_anonymous_names(self, local_anonymous, remainds_indexes):
        if self.is_guest:
            self.transfer_variable.anonymous_guest.remote(
                (local_anonymous, remainds_indexes), role="host"
            )
            (
                remote_anonymous,
                remote_remainds_indexes,
            ) = self.transfer_variable.anonymous_host.get(role="host", idx=0)
        else:
            self.transfer_variable.anonymous_host.remote(
                (local_anonymous, remainds_indexes), role="guest"
            )
            (
                remote_anonymous,
                remote_remainds_indexes,
            ) = self.transfer_variable.anonymous_guest.get(role="guest", idx=0)
        return remote_anonymous, remote_remainds_indexes


class PearsonModelSaver:
    def __init__(self) -> None:
        from federatedml.protobuf.generated import (
            pearson_model_meta_pb2,
            pearson_model_param_pb2,
        )

        self.meta_pb = pearson_model_meta_pb2.PearsonModelMeta()
        self.param_pb = pearson_model_param_pb2.PearsonModelParam()
        self.param_pb.model_name = "HeteroPearson"

    def export(self):
        MODEL_META_NAME = "HeteroPearsonModelMeta"
        MODEL_PARAM_NAME = "HeteroPearsonModelParam"

        return {MODEL_META_NAME: self.meta_pb, MODEL_PARAM_NAME: self.param_pb}

    def save_shapes(self, shapes):
        for shape in shapes:
            self.meta_pb.shapes.append(shape)

    def save_local_corr(self, corr):
        self.param_pb.shape = corr.shape[0]
        for v in corr.reshape(-1):
            self.param_pb.local_corr.append(v.tolist())

    def save_party_info(self, shape, party, names):
        self.param_pb.shapes.append(shape)
        self.param_pb.parties.append(f"({party.role},{party.party_id})")

        _names = self.param_pb.all_names.add()
        for name in names:
            _names.names.append(name)

    def save_local_vif(self, local_vif):
        for vif_value in local_vif:
            self.param_pb.local_vif.append(vif_value)

    def save_cross_corr(self, corr):
        for v in corr.reshape(-1):
            self.param_pb.corr.append(v.tolist())

    def save_party(self, party):
        self.param_pb.party = f"({party.role},{party.party_id})"

    def save_local_anonymous(self, names, anonymous_names):
        for name, anonymous_name in zip(names, anonymous_names):
            self.param_pb.names.append(name)
            anonymous = self.param_pb.anonymous_map.add()
            anonymous.name = name
            anonymous.anonymous = anonymous_name


def standardize(data):
    """
    x -> (x - mu) / sigma
    """
    n = data.count()
    sum_x, sum_square_x = data.mapValues(lambda x: (x, x ** 2)).reduce(
        lambda pair1, pair2: (pair1[0] + pair2[0], pair1[1] + pair2[1])
    )
    mu = sum_x / n
    sigma = np.sqrt(sum_square_x / n - mu ** 2)
    size = len(sigma)
    remiands_indexes = [i for i, e in enumerate(sigma) if e > 0]
    if len(remiands_indexes) < size:
        LOGGER.warning(
            f"zero standard deviation detected, sigma={sigma}, zeroindexes={np.argwhere(sigma)}"
        )
        return (
            n,
            data.mapValues(
                lambda x: (x[remiands_indexes] - mu[remiands_indexes])
                / sigma[remiands_indexes]
            ),
            remiands_indexes,
            size,
        )
    return n, data.mapValues(lambda x: (x - mu) / sigma), remiands_indexes, size


def select_columns(data_instance, hit_column_indexes, hit_column_names):
    """
    select features
    """
    column_names = data_instance.schema["header"]
    num_columns = len(column_names)

    # accept all features
    if hit_column_indexes == -1:
        if len(hit_column_names) > 0:
            raise ValueError(f"specify column name when column_indexes=-1 is ambiguity")
        return column_names, data_instance.mapValues(lambda inst: inst.features)

    # check hit column indexes and column names
    name_to_index = {c: i for i, c in enumerate(column_names)}
    selected = set()
    for name in hit_column_names:
        if name not in name_to_index:
            raise ValueError(f"feature name `{name}` not found in data schema")
        else:
            selected.add(name_to_index[name])
    for idx in hit_column_indexes:
        if 0 <= idx < num_columns:
            selected.add(idx)
        else:
            raise ValueError(f"feature idx={idx} out of bound")
    selected = sorted(list(selected))

    # take shortcut if all feature hit
    if len(selected) == len(column_names):
        return column_names, data_instance.mapValues(lambda inst: inst.features)

    return (
        [column_names[i] for i in selected],
        data_instance.mapValues(lambda inst: inst.features[selected]),
    )


def vif_from_pearson_matrix(pearson_matrix, threshold=1e-8):
    LOGGER.info(f"local vif calc: start")
    assert not np.isnan(
        pearson_matrix
    ).any(), f"should not contains nan: {pearson_matrix}"
    N = pearson_matrix.shape[0]
    vif = []
    LOGGER.info(f"local vif calc: calc matrix eigvals")
    eig = sorted([abs(v) for v in np.linalg.eigvalsh(pearson_matrix)])
    num_drop = len(list(filter(lambda x: x < threshold, eig)))
    det_non_zero = np.prod(eig[num_drop:])
    LOGGER.info(f"local vif calc: calc submatrix eigvals")
    for i in range(N):
        indexes = [j for j in range(N) if j != i]
        cofactor_matrix = pearson_matrix[indexes][:, indexes]
        cofactor_eig = sorted([abs(v) for v in np.linalg.eigvalsh(cofactor_matrix)])
        vif.append(np.prod(cofactor_eig[num_drop:]) / det_non_zero)
        LOGGER.info(f"local vif calc: submatrix {i+1}/{N} eig is {vif[-1]}")
    LOGGER.info(f"local vif calc done")
    return vif


def fix_local_corr(remaind_corr, remainds_indexes, size):
    corr = np.zeros((size, size))
    corr.fill(np.nan)
    corr[np.ix_(remainds_indexes, remainds_indexes)] = np.clip(remaind_corr, -1.0, 1.0)
    return corr


def fix_vif(remains_vif, remainds_indexes, size):
    vif = np.zeros(size)
    vif.fill(np.nan)
    vif[remainds_indexes] = remains_vif
    return vif


def fix_corr(remaind_corr, m1, m2, remainds_indexes1, remainds_indexes2):
    corr = np.zeros((m1, m2))
    corr.fill(np.nan)
    corr[np.ix_(remainds_indexes1, remainds_indexes2)] = np.clip(
        remaind_corr, -1.0, 1.0
    )
    return corr
