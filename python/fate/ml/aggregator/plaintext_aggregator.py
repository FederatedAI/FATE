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

from fate.arch import Context
from fate.ml.aggregator.base import BaseAggregatorClient, BaseAggregatorServer


class PlainTextAggregatorClient(BaseAggregatorClient):
    def __init__(self, ctx: Context, aggregator_name: str = None, aggregate_type="mean", sample_num=1) -> None:
        super().__init__(ctx, aggregator_name, aggregate_type, sample_num, is_mock=True)


class PlainTextAggregatorServer(BaseAggregatorServer):
    def __init__(self, ctx: Context, aggregator_name: str = None) -> None:
        super().__init__(ctx, aggregator_name, is_mock=True)
