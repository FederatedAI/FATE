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
from typing import Optional, Union

from fate.arch import Context
from fate.arch.dataframe import DataFrame


class Model:
    ...


class Module:
    mode: str

    def fit(
        self,
        ctx: Context,
        train_data: DataFrame,
        validate_data: Optional[DataFrame] = None,
    ) -> None:
        ...

    def transform(self, ctx: Context, transform_data: DataFrame) -> DataFrame:
        ...

    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        ...

    def from_model(cls, model: Union[dict, Model]):
        ...

    def get_model(self) -> Union[dict, Model]:
        ...


class HomoModule(Module):
    mode = "HOMO"


class HeteroModule(Module):
    mode = "HETERO"
