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
import time
import random
import hashlib


def generate_sample_id(n, prefix):
    return [hashlib.sha256(bytes(prefix + str(i), encoding="utf-8")).hexdigest() for i in range(n)]


def generate_sample_id_prefix():
    return str(time.time()) + str(random.randint(1000000, 9999999))
