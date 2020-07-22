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
from fate_flow.entity.constant import BaseJobStatus, InterruptStatus, EndStatus


class StatusEngine(object):
    @staticmethod
    def vertical_convergence(downstream_status_list):
        tmp_status_set = set(downstream_status_list)
        if len(tmp_status_set) == 1:
            return tmp_status_set.pop()
        else:
            for interrupt_status in InterruptStatus.status_list():
                if interrupt_status in tmp_status_set:
                    return interrupt_status
            else:
                if BaseJobStatus.RUNNING in tmp_status_set:
                    return BaseJobStatus.RUNNING
                elif BaseJobStatus.WAITING in tmp_status_set:
                    return BaseJobStatus.WAITING
                else:
                    raise Exception("The list of vertically convergent status failed: {}".format(",".join(downstream_status_list)))

    @staticmethod
    def horizontal_convergence(floor_status_list):
        tmp_status_set = set(floor_status_list)
        if len(tmp_status_set) == 1:
            return tmp_status_set.pop()
        else:
            for interrupt_status in InterruptStatus.status_list():
                if interrupt_status in tmp_status_set:
                    return interrupt_status
            else:
                if BaseJobStatus.RUNNING in tmp_status_set:
                    return BaseJobStatus.RUNNING
                elif BaseJobStatus.WAITING in tmp_status_set:
                    return BaseJobStatus.WAITING
                else:
                    raise Exception("The list of horizontal convergent status failed: {}".format(",".join(floor_status_list)))
