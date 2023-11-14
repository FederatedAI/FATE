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

import torch
from ..abc.module import Module
from fate.arch import Context
from fate.arch.dataframe import DataFrame


class PearsonCorrelation(Module):
    def __init__(self):
        ...

    def fit(self, ctx: Context, input_data: DataFrame):
        # standardize data
        ctx.mpc.init()
        data = input_data.as_tensor()
        n = data.shape[0]
        data_mean = torch.mean(data, dim=0)
        data_std = torch.std(data, dim=0)
        data = (data - data_mean) / data_std
        rank_a, rank_b = ctx.guest.rank, ctx.hosts[0].rank
        with ctx.mpc.communicator.new_group(ranks=[rank_a, rank_b], name="pearson_correlation"):
            x = ctx.mpc.lazy_encrypt(lambda: data.T, src=rank_a)
            y = ctx.mpc.lazy_encrypt(lambda: data, src=rank_b)
            out = x.matmul(y).get_plain_text()
            ctx.mpc.info(f"pearson correlation={out / n}")
