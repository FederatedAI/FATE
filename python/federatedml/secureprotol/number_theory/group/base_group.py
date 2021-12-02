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


class GroupElement(object):
    """
    Group element
    """

    def __init__(self):
        pass


class GroupArithmetic(object):
    """
    A collection of arithmetic operators for groups
    """

    def __init__(self, identity):
        self.identity = identity

    def add(self, a, b):
        """
        x + y
        """

    def neg(self, a):
        """
        -x
        """
        pass

    def sub(self, a, b):
        """
        x - y
        """
        pass

    def mul(self, scalar, a):
        """
        scalar * a
        """
        pass

    def get_identity(self):
        return self.identity
