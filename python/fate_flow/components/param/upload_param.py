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
from fate_arch.storage import DEFAULT_ID_DELIMITER


class UploadParam:
    def __init__(self, file="", head=1, id_delimiter=DEFAULT_ID_DELIMITER, partition=10, namespace="", name="",
                 storage_engine="", storage_address=None, destroy=False, extend_sid=False, auto_increasing_sid=False):
        self.file = file
        self.head = head
        self.id_delimiter = id_delimiter
        self.partition = partition
        self.namespace = namespace
        self.name = name
        self.storage_engine = storage_engine
        self.storage_address = storage_address
        self.destroy = destroy
        self.extend_sid = extend_sid
        self.auto_increasing_sid = auto_increasing_sid

    def check(self):
        return True

