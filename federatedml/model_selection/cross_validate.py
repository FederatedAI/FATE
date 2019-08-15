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
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class BaseCrossValidator(object):

    def __init__(self):
        self.mode = None
        self.role = None

    def split(self, data_inst): pass

    def display_cv_result(self, cv_results):
        LOGGER.debug("cv_result: {}".format(cv_results))
        if self.role == consts.GUEST or (self.role == consts.HOST and self.mode == consts.HOMO):
            format_cv_result = {}
            for eval_result in cv_results:
                for eval_name, eval_r in eval_result.items():
                    if not isinstance(eval_r, list):
                        if eval_name not in format_cv_result:
                            format_cv_result[eval_name] = []
                        format_cv_result[eval_name].append(eval_r)
                    else:
                        for e_r in eval_r:
                            e_name = "{}_thres_{}".format(eval_name, e_r[0])
                            if e_name not in format_cv_result:
                                format_cv_result[e_name] = []
                            format_cv_result[e_name].append(e_r[1])

            for eval_name, eva_result_list in format_cv_result.items():
                mean_value = np.around(np.mean(eva_result_list), 4)
                std_value = np.around(np.std(eva_result_list), 4)
                LOGGER.info("{}，evaluate name: {}, mean: {}, std: {}".format(self.role,
                                                                             eval_name, mean_value, std_value))
