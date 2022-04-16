
"""
This module provide some utilized methods that operate the index of distributed data
"""


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

def collect_index(data_insts):
    data_sids = data_insts.mapValues(lambda data_inst: None)
    # data_size = data_sids.count()  # Record data nums that left
    data_sids_iter = data_sids.collect()
    data_sids_iter = sorted(data_sids_iter, key=lambda x: x[0])
    data_size = len(data_sids_iter)
    return data_sids_iter, data_size
