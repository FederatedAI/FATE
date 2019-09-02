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

from federatedml.util import consts


class Guest(object):
    def _register_intermediate_transfer(self, *transfervariables):
        raise NotImplementedError("Method not Implemented")

    def host_to_guest(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.HOST,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars

    def guest_to_host(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.HOST,
                                     idx=idx,
                                     suffix=suffix)

    def guest_to_arbiter(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.ARBITER,
                                     idx=idx,
                                     suffix=suffix)

    def arbiter_to_guest(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.ARBITER,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars


class Host(object):
    def _register_intermediate_transfer(self, *transfervariables):
        raise NotImplementedError("Method not Implemented")

    def guest_to_host(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.GUEST,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars

    def host_to_guest(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.GUEST,
                                     idx=idx,
                                     suffix=suffix)

    def host_to_arbiter(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.ARBITER,
                                     idx=idx,
                                     suffix=suffix)

    def arbiter_to_host(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.ARBITER,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars


class Arbiter(object):
    def _register_intermediate_transfer(self, *transfervariables):
        raise NotImplementedError("Method not Implemented")

    def guest_to_arbiter(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.GUEST,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars

    def host_to_arbiter(self, transfer_variables, idx=0, suffix=tuple()):
        received_vars = []
        for transfer_variable in transfer_variables:
            this_var = transfer_variable.get(role=consts.HOST,
                                             idx=idx,
                                             suffix=suffix)
            received_vars.append(this_var)
        return received_vars

    def arbiter_to_host(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.HOST,
                                     idx=idx,
                                     suffix=suffix)

    def arbiter_to_guest(self, variables, transfer_variables, idx=0, suffix=tuple()):
        for variable, transfer_variable in zip(variables, transfer_variables):
            transfer_variable.remote(obj=variable,
                                     role=consts.GUEST,
                                     idx=idx,
                                     suffix=suffix)
