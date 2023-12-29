#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import logging

import pandas as pd
import torch

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from ..abc.module import Module

logger = logging.getLogger(__name__)


class PearsonCorrelation(Module):
    def __init__(self, local_only=False, calc_local_vif=True, select_cols=None):
        self.local_only = local_only
        self.calc_local_vif = calc_local_vif
        self.select_cols = select_cols
        self.local_corr = None
        self.remote_corr = None
        self.vif = None
        self.select_anonymous_cols = None

    def fit(self, ctx: Context, input_data: DataFrame):
        if len(ctx.hosts) > 1:
            raise ValueError(f"feature correlation only support one host, but {len(ctx.hosts)} hosts exist.")
        # standardize data
        ctx.mpc.init()
        if self.select_cols is None:
            self.select_cols = input_data.schema.columns
        to_compute_data = input_data[self.select_cols]
        data = to_compute_data.as_tensor()
        n = data.shape[0]
        num_features = data.shape[1]
        data_mean = torch.mean(data, dim=0)
        data_std = torch.std(data, dim=0)
        data = (data - data_mean) / data_std
        local_corr = torch.matmul(data.T, data) / (n - 1)
        # remainds_index = [i for i in range(data_std.shape[0]) if data_std[i] > 0]
        self.local_corr = pd.DataFrame(local_corr, columns=self.select_cols, index=self.select_cols)

        if self.calc_local_vif:
            logger.info("calc_local_vif enabled, calculate vif for local features")
            local_vif = self.vif_from_pearson_matrix(local_corr)
            # fixed_local_vif = self.fix_vif(local_vif, remainds_index, num_features)
            # self.vif = fixed_local_vif
            local_vif_list = [vif.item() for vif in local_vif]
            self.vif = pd.DataFrame(local_vif_list, columns=["vif"], index=self.select_cols)
        else:
            logger.info("calc_local_vif disabled, skip local vif")

        if not self.local_only:
            rank_a, rank_b = ctx.guest.rank, ctx.hosts[0].rank
            anonymous_header = [
                input_data.schema.anonymous_columns[input_data.schema.columns.get_loc(col)] for col in self.select_cols
            ]
            self.select_anonymous_cols = anonymous_header
            if ctx.is_on_guest:
                ctx.hosts.put("anonymous_header", anonymous_header)
                a_header = input_data.schema.columns
                b_header = ctx.hosts.get("anonymous_header")[0]
            else:
                ctx.guest.put("anonymous_header", anonymous_header)
                a_header = ctx.guest.get("anonymous_header")
                b_header = input_data.schema.columns

            with ctx.mpc.communicator.new_group(ranks=[rank_a, rank_b], name="pearson_correlation"):
                x = ctx.mpc.lazy_encrypt(lambda: data.T, src=rank_a)
                y = ctx.mpc.lazy_encrypt(lambda: data, src=rank_b)
                remote_corr = x.matmul(y).get_plain_text() / (n - 1)
                self.remote_corr = pd.DataFrame(remote_corr, columns=b_header, index=a_header)
                # ctx.mpc.info(f"pearson correlation={out / n}")

    @staticmethod
    def vif_from_pearson_matrix(pearson_matrix, threshold=1e-8):
        logger.info(f"local vif calc: start")
        assert not torch.isnan(pearson_matrix).any(), f"should not contains nan: {pearson_matrix}"
        N = pearson_matrix.shape[0]
        vif = []
        logger.info(f"local vif calc: calc matrix eigvals")
        eig, _ = torch.sort(torch.abs(torch.linalg.eigvalsh(pearson_matrix)))
        num_drop = torch.sum(eig < threshold).item()
        det_non_zero = torch.prod(eig[num_drop:])
        logger.info(f"local vif calc: calc submatrix eigvals")
        for i in range(N):
            indexes = [j for j in range(N) if j != i]
            cofactor_matrix = pearson_matrix[indexes][:, indexes]
            cofactor_eig, _ = torch.sort(torch.abs(torch.linalg.eigvalsh(cofactor_matrix)))
            vif.append(torch.prod(cofactor_eig[num_drop:]) / det_non_zero)
            logger.info(f"local vif calc: submatrix {i + 1}/{N} eig is {vif[-1]}")
        logger.info(f"local vif calc done")
        return vif

    """@staticmethod
    def fix_vif(remainds_vif, remainds_indexes, size):
        vif = torch.empty(size).fill_(float("nan"))
        remainds_mask = torch.tensor([i in remainds_indexes for i in range(size)])
        vif[remainds_mask] = torch.tensor(remainds_vif)
        return vif"""

    def get_model(self):
        output_model = {
            "data": {
                "local_corr": self.local_corr.to_dict() if self.local_corr is not None else None,
                "remote_corr": self.remote_corr.to_dict() if self.remote_corr is not None else None,
                "vif": self.vif.to_dict() if self.vif is not None else None,
            },
            "meta": {
                "model_type": "feature_correlation",
                "column_anonymous_map": dict(zip(self.select_cols, self.select_anonymous_cols))
                if self.select_anonymous_cols
                else None,
            },
        }
        return output_model
