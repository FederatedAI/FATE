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

import numpy as np
from arch.api import eggroll
# from arch.api.utils import log_utils

from sklearn.model_selection import KFold as sk_KFold
from federatedml.model_selection.indices import collect_index
from federatedml.model_selection.cross_validate import BaseCrossValidator

# LOGGER = log_utils.getLogger()


class KFold(BaseCrossValidator):
    def __init__(self, n_splits=5):
        super(KFold, self).__init__()
        self.n_splits = n_splits

    def split(self, data_inst, shuffle=True):
        header = data_inst.schema.get('header')

        data_sids_iter, data_size = collect_index(data_inst)

        data_sids = []
        for sid, _ in data_sids_iter:
            data_sids.append(sid)
        data_sids = np.array(data_sids)

        if shuffle:
            np.random.shuffle(data_sids)

        kf = sk_KFold(n_splits=self.n_splits)

        for train, test in kf.split(data_sids):
            train_sids = data_sids[train]
            test_sids = data_sids[test]
            train_sids_table = [(str(x), 1) for x in train_sids]
            test_sids_table = [(str(x), 1) for x in test_sids]
            # print(train_sids_table)
            train_table = eggroll.parallelize(train_sids_table,
                                              include_key=True,
                                              partition=data_inst._partitions)
            train_data = data_inst.join(train_table, lambda x, y: x)
            test_table = eggroll.parallelize(test_sids_table,
                                             include_key=True,
                                             partition=data_inst._partitions)
            test_data = data_inst.join(test_table, lambda x, y: x)
            train_data.schema['header'] = header
            test_data.schema['header'] = header
            yield train_data, test_data
