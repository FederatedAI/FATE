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
from fate_flow.entity.types import RunParameters


class JobRuntimeConfigAdapter(object):
    def __init__(self, job_runtime_conf):
        self.job_runtime_conf = job_runtime_conf

    def get_common_parameters(self):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            if 'common' in self.job_runtime_conf['job_parameters']:
                job_parameters = RunParameters(**self.job_runtime_conf['job_parameters']['common'])
            else:
                job_parameters = RunParameters(**self.job_runtime_conf['job_parameters'])
            self.job_runtime_conf['job_parameters']['common'] = job_parameters.to_dict()
        else:
            job_parameters = RunParameters(**self.job_runtime_conf['job_parameters'])
            self.job_runtime_conf['job_parameters'] = job_parameters.to_dict()
        return job_parameters

    def get_job_parameters_dict(self, job_parameters=None):
        if job_parameters:
            if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
                self.job_runtime_conf['job_parameters']['common'] = job_parameters.to_dict()
            else:
                self.job_runtime_conf['job_parameters'] = job_parameters.to_dict()
        return self.job_runtime_conf['job_parameters']

    def get_job_work_mode(self):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            work_mode = self.job_runtime_conf['job_parameters'].get('common', {}).get('work_mode')
        else:
            work_mode = self.job_runtime_conf['job_parameters'].get('work_mode')
        return work_mode




