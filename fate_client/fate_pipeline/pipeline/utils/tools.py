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


def merge_dict(dict1, dict2):
    merge_ret = {}
    keyset = dict1.keys() | dict2.keys()
    for key in keyset:
        if key in dict1 and key in dict2:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            assert type(val1).__name__ == type(val2).__name__
            if isinstance(val1, dict):
                merge_ret[key] = merge_dict(val1, val2)
            else:
                merge_ret[key] = val2
        elif key in dict1:
            merge_ret[key] = dict1.get(key)
        else:
            merge_ret[key] = dict2.get(key)

    return merge_ret


def extract_explicit_parameter(func):
    def wrapper(*args, **kwargs):
        explict_kwargs = {"explict_parameters": kwargs}

        return func(*args, **explict_kwargs)

    return wrapper
