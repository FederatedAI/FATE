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
            if "common" not in self.job_runtime_conf["job_parameters"]:
                raise RuntimeError(
                    "the configuration format for v2 version must be job_parameters:common")
            job_parameters = RunParameters(
                **self.job_runtime_conf['job_parameters']['common'])
            self.job_runtime_conf['job_parameters']['common'] = job_parameters.to_dict(
            )
        else:
            if "processors_per_node" in self.job_runtime_conf['job_parameters']:
                self.job_runtime_conf['job_parameters']["eggroll_run"] = \
                    {"eggroll.session.processors.per.node":
                        self.job_runtime_conf['job_parameters']["processors_per_node"]}
            job_parameters = RunParameters(
                **self.job_runtime_conf['job_parameters'])
            self.job_runtime_conf['job_parameters'] = job_parameters.to_dict()
        return job_parameters

    def update_common_parameters(self, common_parameters: RunParameters):
        if int(self.job_runtime_conf.get("dsl_version", 1)) == 2:
            if "common" not in self.job_runtime_conf["job_parameters"]:
                raise RuntimeError(
                    "the configuration format for v2 version must be job_parameters:common")
            self.job_runtime_conf["job_parameters"]["common"] = common_parameters.to_dict(
            )
        else:
            self.job_runtime_conf["job_parameters"] = common_parameters.to_dict(
            )
        return self.job_runtime_conf

    def get_job_parameters_dict(self, job_parameters: RunParameters = None):
        if job_parameters:
            if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
                self.job_runtime_conf['job_parameters']['common'] = job_parameters.to_dict(
                )
            else:
                self.job_runtime_conf['job_parameters'] = job_parameters.to_dict(
                )
        return self.job_runtime_conf['job_parameters']

    def get_job_work_mode(self):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            work_mode = self.job_runtime_conf['job_parameters'].get(
                'common', {}).get('work_mode')
        else:
            work_mode = self.job_runtime_conf['job_parameters'].get(
                'work_mode')
        return work_mode

    def get_job_backend(self):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            backend = self.job_runtime_conf['job_parameters'].get(
                'common', {}).get('backend')
        else:
            backend = self.job_runtime_conf['job_parameters'].get('backend')
        return backend

    def get_job_type(self):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            job_type = self.job_runtime_conf['job_parameters'].get(
                'common', {}).get('job_type')
            if not job_type:
                job_type = self.job_runtime_conf['job_parameters'].get(
                    'job_type', 'train')
        else:
            job_type = self.job_runtime_conf['job_parameters'].get(
                'job_type', 'train')
        return job_type

    def update_model_id_version(self, model_id=None, model_version=None):
        if int(self.job_runtime_conf.get('dsl_version', 1)) == 2:
            if model_id:
                self.job_runtime_conf['job_parameters'].get(
                    'common', {})['model_id'] = model_id
            if model_version:
                self.job_runtime_conf['job_parameters'].get(
                    'common', {})['model_version'] = model_version
        else:
            if model_id:
                self.job_runtime_conf['job_parameters']['model_id'] = model_id
            if model_version:
                self.job_runtime_conf['job_parameters']['model_version'] = model_version
        return self.job_runtime_conf
