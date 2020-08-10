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


class StoreTypes(object):
    ROLLPAIR_IN_MEMORY = 'IN_MEMORY'
    ROLLPAIR_LMDB = 'LMDB'
    ROLLPAIR_LEVELDB = 'LEVEL_DB'
    ROLLFRAME_FILE = 'ROLL_FRAME_FILE'
    ROLLPAIR_ROLLSITE = 'ROLL_SITE'
    ROLLPAIR_FILE = 'ROLL_PAIR_FILE'
    ROLLPAIR_MMAP = 'ROLL_PAIR_MMAP'
    ROLLPAIR_CACHE = 'ROLL_PAIR_CACHE'
    ROLLPAIR_QUEUE = 'ROLL_PAIR_QUEUE'
