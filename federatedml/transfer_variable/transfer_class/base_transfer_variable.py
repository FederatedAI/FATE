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
from arch.api.utils import log_utils
from arch.api import federation

LOGGER = log_utils.getLogger()


class Variable(object):
    def __init__(self, name, auth, transfer_variable):
        self.name = name
        self.auth = auth
        self._transfer_variable = transfer_variable

    def remote(self, obj, role=None, idx=-1, suffix=tuple()):
        """
        send obj to other parties.

        Args:
            obj: object to be sent
            role: role of parties to sent to, use one of ['Host', 'Guest', 'Arbiter', None].
                The default is None, means sent values to parties regardless their party role
            idx: id of party to sent to.
                The default is -1, which means sent values to parties regardless their party id
            suffix: additional tag suffix, the default is tuple()
        """
        if not isinstance(suffix, tuple):
            suffix = (suffix, )
        federation.remote(obj=obj,
                          name=self.name,
                          tag=self._transfer_variable.generate_transferid(self, *suffix),
                          role=role,
                          idx=idx)

    def get(self, idx=-1, suffix=tuple()):
        """
        get obj from other parties.

        Args:
            idx: id of party to get from.
                The default is -1, which means get values from parties regardless their party id
            suffix: additional tag suffix, the default is tuple()

        Returns:
            object or list of object
        """
        if not isinstance(suffix, tuple):
            suffix = (suffix, )
        tag = self._transfer_variable.generate_transferid(self, *suffix)
        name = self.name
        if isinstance(idx, int):
            return federation.get(name=name, tag=tag, idx=idx)
        elif isinstance(idx, list):
            name = self.name
            return [federation.get(name=name, tag=tag, idx=pid) for pid in idx]
        else:
            raise ValueError(f"illegal idx type: {type(idx)}, supported types: int or list of int")


class BaseTransferVariable(object):
    def __init__(self, flowid=0):
        self.flowid = flowid
        self.define_transfer_variable()

    def set_flowid(self, flowid):
        self.flowid = flowid

    def set_taskid(self, taskid):
        self.taskid = taskid

    def generate_transferid(self, transfer_var, *suffix):
        if transfer_var.name.split(".", -1)[-1] not in self.__dict__:
            raise ValueError("transfer variable not in class, please check if!!!")

        #transferid = transfer_var.name + "." + str(self.flowid) + "." + str(self.taskid)
        transferid = transfer_var.name + "." + str(self.flowid)
        if suffix:
            transferid += "." + ".".join(map(str, suffix))
        # LOGGER.debug("transferid is :{}, taskid is : {}".format(transferid, self.taskid))
        return transferid

    def define_transfer_variable(self):
        pass
