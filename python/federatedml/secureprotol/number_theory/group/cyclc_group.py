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
from federatedml.secureprotol.number_theory.group.base_group import GroupElement, GroupArithmetic


class CyclicGroupElement(GroupElement):
    """
    Cyclic group element
    """

    def __init__(self):
        super(CyclicGroupElement, self).__init__()


class CyclicGroupArithmetic(GroupArithmetic):
    """
    A collection of arithmetic operators for cyclic groups
    """

    def __init__(self, identity, generator):
        super(CyclicGroupArithmetic, self).__init__(identity)
        self.generator = generator

    def get_generator(self):
        return self.generator
