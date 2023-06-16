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
import pandas as pd
from typing import List


def series_to_ndarray(series_obj: "pd.Series", fields_to_align: List[str]=None):
    if isinstance(series_obj.index, pd.RangeIndex) or not fields_to_align:
        return series_obj.values
    else:
        if len(series_obj) != len(fields_to_align):
            raise ValueError(f"Can't not align fields, src={fields_to_align}, dst={series_obj}")

        indexer = series_obj.index.get_indexer(fields_to_align)

        return series_obj[indexer].values


def series_to_list(series_obj: "pd.Series", fields_to_align: List[str]=None):
    return series_to_ndarray(series_obj, fields_to_align).tolist()
