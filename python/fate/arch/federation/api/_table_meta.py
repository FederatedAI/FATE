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


class TableMeta:
    def __init__(self, num_partitions: int, key_serdes_type: int, value_serdes_type: int, partitioner_type: int):
        self.num_partitions = num_partitions
        self.key_serdes_type = key_serdes_type
        self.value_serdes_type = value_serdes_type
        self.partitioner_type = partitioner_type
