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
from fate_flow.entity.constant import StatusSet, OngoingStatus, InterruptStatus, EndStatus


class StatusEngine(object):
    @staticmethod
    def vertical_convergence(downstream_status_list, interrupt_break=True):
        tmp_status_set = set(downstream_status_list)
        if len(tmp_status_set) == 1:
            return tmp_status_set.pop()
        else:
            if interrupt_break:
                for status in sorted(InterruptStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                    if status in tmp_status_set:
                        return status
            else:
                # Check to if all status are end status
                for status in tmp_status_set:
                    if not EndStatus.contains(status=status):
                        break
                else:
                    # All status are end status and there are more than two different status
                    for status in sorted(InterruptStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                        if status in tmp_status_set:
                            return status
                    raise Exception("The list of vertically convergent status failed: {}".format(downstream_status_list))
            # All ongoing status or ongoing status + complete status
            if StatusSet.COMPLETE in tmp_status_set:
                return StatusSet.RUNNING
            for status in sorted(OngoingStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status in tmp_status_set:
                    return status
            raise Exception("The list of vertically convergent status failed: {}".format(downstream_status_list))

    @staticmethod
    def horizontal_convergence(floor_status_list):
        tmp_status_set = set(floor_status_list)
        if len(tmp_status_set) == 1:
            return tmp_status_set.pop()
        else:
            for status in sorted(InterruptStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status in tmp_status_set:
                    return status
            # All ongoing status or ongoing status + complete status
            if StatusSet.COMPLETE in tmp_status_set:
                return StatusSet.RUNNING
            for status in sorted(OngoingStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status in tmp_status_set:
                    return status
            raise Exception("The list of horizontal convergent status failed: {}".format(",".join(floor_status_list)))
