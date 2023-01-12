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
import time
import random
import uuid


def get_uuid() -> str:
    return str(uuid.uuid1())


def time_str() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def gen_job_id(*suffix) -> str:
    if not suffix:
        return time_str() + str(random.randint(0, 100000))
    else:
        return "_".join(list(suffix))


def gen_computing_id(job_id, task_name, role, party_id) -> str:
    return "_".join([job_id, task_name, role, party_id, "computing"])


def gen_task_id(job_id, task_name, role, party_id) -> str:
    return "_".join([job_id, task_name, role, party_id, "execution"])


def gen_federation_id(job_id, task_name) -> str:
    return "_".join([job_id, task_name, "federation"])
