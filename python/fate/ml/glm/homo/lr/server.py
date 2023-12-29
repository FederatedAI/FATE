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

from fate.ml.abc.module import HomoModule
from fate.arch.dataframe import DataFrame
from fate.arch import Context
import logging
from fate.ml.nn.homo.fedavg import FedAVGServer


logger = logging.getLogger(__name__)


class HomoLRServer(HomoModule):
    def __init__(self) -> None:
        pass

    def fit(self, ctx: Context, data: DataFrame = None) -> None:
        server = FedAVGServer(ctx=ctx)
        logger.info("server class init done, start fed training")
        server.train()
        logger.info("homo lr fit done")

    def predict(self, ctx: Context, predict_data: DataFrame = None) -> DataFrame:
        logger.info("skip prediction stage")
