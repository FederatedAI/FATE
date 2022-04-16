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


import io
import sys
import json


# Datastream is a wraper of StringIO, it receives kv pairs and dump it to json string
class Datastream(object):
    def __init__(self):
        self._string = io.StringIO()
        self._string.write("[")

    def get_size(self):
        return sys.getsizeof(self._string.getvalue())

    def get_data(self):
        self._string.write("]")
        return self._string.getvalue()

    def append(self, kv: dict):
        # add ',' if not the first element
        if self._string.getvalue() != "[":
            self._string.write(",")
        json.dump(kv, self._string)

    def clear(self):
        self._string.close()
        self.__init__()
