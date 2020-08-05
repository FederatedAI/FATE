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

from fate_arch.federation.transfer_variable import BaseTransferVariables
from fate_arch.federation.transfer_variable._generated_enhance_variable import *


class EnhanceTransferVariables(BaseTransferVariables):
    def __init__(self, flow_id=0):
        super().__init__(flow_id)
        base_cls = f"{self.__class__.__module__}${self.__class__.__name__}"
        for k, cls in getattr(self, "__annotations__", {}).items():
            if not issubclass(cls, (
                    A2GVariable, A2HVariable, G2AVariable, G2HVariable, H2AVariable, H2GVariable,
                    A2GHVariable, G2AHVariable, H2AGVariable, AG2HVariable, AH2GVariable, GH2AVariable)):
                continue
            name = f"{base_cls}${k}"
            obj = cls.get_or_create(name, lambda: cls(name))
            setattr(self, k, obj)


__all__ = ["EnhanceTransferVariables",
           "A2GVariable", "A2HVariable", "G2AVariable", "G2HVariable", "H2AVariable", "H2GVariable",
           "A2GHVariable", "G2AHVariable", "H2AGVariable", "AG2HVariable", "AH2GVariable", "GH2AVariable"]
