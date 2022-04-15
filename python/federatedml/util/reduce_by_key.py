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


def reduce(table, func, key_func=None):
    if key_func is None:
        return table.reduce(func)

    it = table.collect()
    ret = {}
    for k, v in it:
        agg_key = key_func(k)
        if agg_key in ret:
            ret[agg_key] = func(ret[agg_key], v)
        else:
            ret[agg_key] = v
    return ret
