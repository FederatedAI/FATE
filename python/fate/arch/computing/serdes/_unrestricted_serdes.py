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

import os

from ._serdes_base import p_dumps, p_loads


def get_unrestricted_serdes():
    if True or os.environ.get("SERDES_DEBUG_MODE") == "1":
        return UnrestrictedSerdes
    else:
        raise PermissionError("UnsafeSerdes is not allowed in production mode")


class UnrestrictedSerdes:
    @staticmethod
    def serialize(obj) -> bytes:
        return p_dumps(obj)

    @staticmethod
    def deserialize(bytes) -> object:
        return p_loads(bytes)
