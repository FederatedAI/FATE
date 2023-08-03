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

from fate.arch import Context
from fate.arch.dataframe import utils
from ..abc.module import Module

logger = logging.getLogger(__name__)


class SampleModuleGuest(Module):
    def __init__(
            self,
            replace=False,
            frac=1.0,
            n=None,
            random_state=None,
            hetero_sync=True
    ):
        self.replace = replace
        self.frac = frac
        self.n = n
        self.random_state = random_state
        self.hetero_sync = hetero_sync

        self._sample_obj = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info(f"enter sample fit")
        if self.hetero_sync:
            logger.info(f"hetero sync")
            logger.info(f"role: {ctx.local.role}")

            sampled_data = utils.federated_sample(ctx,
                                                  train_data,
                                                  n=self.n,
                                                  frac=self.frac,
                                                  replace=self.replace,
                                                  role=ctx.local.role,
                                                  random_state=self.random_state)
        else:
            logger.info(f"local sample")
            # local sample
            sampled_data = utils.local_sample(ctx,
                                              train_data,
                                              n=self.n,
                                              frac=self.frac,
                                              replace=self.replace,
                                              random_state=self.random_state)

        return sampled_data


class SampleModuleHost(Module):
    def __init__(
            self,
            replace=False,
            frac=1.0,
            n=None,
            random_state=None,
            hetero_sync=True
    ):
        self.replace = replace
        self.frac = frac
        self.n = n
        self.random_state = random_state
        self.hetero_sync = hetero_sync

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info(f"enter sample fit")
        if self.hetero_sync:
            logger.info(f"hetero sync")
            logger.info(f"role: {ctx.local.role}")

            sampled_data = utils.federated_sample(ctx,
                                                  train_data,
                                                  role=ctx.local.role)
        else:
            # local sample
            logger.info(f"local sample")
            sampled_data = utils.local_sample(ctx,
                                              train_data,
                                              n=self.n,
                                              frac=self.frac,
                                              replace=self.replace,
                                              random_state=self.random_state)
            """elif self.mode == "weight":
                if self.n is not None:
                    sampled_data = train_data.sample(n=self.n,
                                                     replace=self.replace,
                                                     weight=train_data.weight,
                                                     random_state=self.random_state)
                else:
                    sampled_data = train_data.sample(frac=self.frac,
                                                     relace=self.replace,
                                                     weight=train_data.weight,
                                                     random_state=self.random_state)"""

        return sampled_data
