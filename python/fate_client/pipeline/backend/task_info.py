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

import sys

from pipeline.utils.logger import LOGGER


class TaskInfo(object):
    def __init__(self, jobid, component, job_client, role='guest', party_id=9999):
        self._jobid = jobid
        self._component = component
        self._job_client = job_client
        self._party_id = party_id
        self._role = role

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_output_data(self, limits=None):
        '''
        gets downloaded data of arbitrary component
        Parameters
        ----------
        limits: int, None, default None. Maximum number of lines returned, including header. If None, return all lines.

        Returns
        -------
        dict
        single output example:
            {
                data: [],
                meta: []

            }
        multiple output example:
            {
            train_data: {
                data: [],
                meta: []
                },
            validate_data: {
                data: [],
                meta: []
                }
            test_data: {
                data: [],
                meta: []
                }
            }
        '''
        return self._job_client.get_output_data(self._jobid, self._component.name, self._role, self._party_id, limits)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_model_param(self):
        '''
        get fitted model parameters
        Returns
        -------
        dict
        '''
        return self._job_client.get_model_param(self._jobid, self._component.name, self._role, self._party_id)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_output_data_table(self):
        '''
        get output data table information, including table name and namespace, as given by flow client
        Returns
        -------
        dict
        '''
        return self._job_client.get_output_data_table(self._jobid, self._component.name, self._role, self._party_id)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_summary(self):
        '''
        get module summary of arbitrary component
        Returns
        -------
        dict
        '''
        return self._job_client.get_summary(self._jobid, self._component.name, self._role, self._party_id)
