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


def stat_method(df, stat_func, *args, index=None, **kwargs) -> pd.Series:
    if "axis" not in kwargs:
        kwargs["axis"] = 0
    stat_ret = getattr(df, stat_func)(*args, **kwargs)
    dtype = str(stat_ret.dtype.to_torch_dtype()).split(".", -1)[-1]
    stat_ret = stat_ret.tolist()
    if not kwargs.get("axis", 0):
        if index:
            return pd.Series(stat_ret, index=index, dtype=dtype)
        else:
            return pd.Series(stat_ret, dtype=dtype)
    else:
        return pd.Series(stat_ret, dtype=dtype)
