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


# from arch.api.utils import log_utils

# LOGGER = log_utils.getLogger()


class Gradient:
    def compute(self, values, coef, intercept, fit_intercept):
        raise NotImplementedError("Method not implemented")

    def compute_loss(self, X, Y, coef, intercept):
        raise NotImplementedError("Method not implemented")

    def load_data(self, data_instance):
        # LOGGER.debug("In load_data of gradient function")
        X = []
        Y = []
        # 获取batch数据
        for iter_key, instant in data_instance:
            weighted_feature = instant.weight * instant.features
            X.append(weighted_feature)
            if instant.label == 1:
                Y.append([1])
            else:
                Y.append([-1])
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
