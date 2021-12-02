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
import hmac

from federatedml.secureprotol.random_oracle.message_authentication_code.hash_mac import \
    HashBasedMessageAuthenticationCode


class Sha256MAC(HashBasedMessageAuthenticationCode):
    """
    Sha256-based MAC
    """

    def __init__(self, key):
        self.mode = 'sha256'
        self.key = key
        super(Sha256MAC, self).__init__(hmac.new(self.key, digestmod=self.mode))

    def digest(self, message):
        self.function = hmac.new(self.key, digestmod=self.mode)
        self.function.update(message)
        return self.function.digest()
