#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
################################################################################

from arch.api.utils import log_utils
from federatedml.param.base_param import BaseParam

LOGGER = log_utils.getLogger()


class PredictParam(BaseParam):
    """
    Define the predict method of HomoLR, HeteroLR, SecureBoosting

    Parameters
    ----------

    threshold: float or int, The threshold use to separate positive and negative class. Normally, it should be (0,1)
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def check(self):

        if type(self.threshold).__name__ not in ["float", "int"]:
            raise ValueError("predict param's predict_param {} not supported, should be float or int".format(
                self.threshold))

        LOGGER.debug("Finish predict parameter check!")
        return True
