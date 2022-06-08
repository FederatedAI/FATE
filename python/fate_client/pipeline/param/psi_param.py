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


from pipeline.param.base_param import BaseParam


class PSIParam(BaseParam):

    def __init__(self, max_bin_num=20, need_run=True, dense_missing_val=None):
        super(PSIParam, self).__init__()
        self.max_bin_num = max_bin_num
        self.need_run = need_run
        self.dense_missing_val = dense_missing_val

    def check(self):
        assert isinstance(self.max_bin_num, int) and self.max_bin_num > 0, 'max bin must be an integer larger than 0'
        assert isinstance(self.need_run, bool)

        if self.dense_missing_val is not None:
            assert isinstance(self.dense_missing_val, str) or isinstance(self.dense_missing_val, int) or \
                isinstance(self.dense_missing_val, float), \
                'missing value type {} not supported'.format(type(self.dense_missing_val))
