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

from arch.api import session
from arch.api import federation

if __name__ == '__main__':
    session.init(job_id="atest")
    federation.init("atest", {

        "local": {
            "role": "guest",
            "party_id": 10001
        },
        "role": {
            "host": [
                10001,
                10002
            ],
            "arbiter": [
                99999
            ],
            "guest": [
                10001
            ]
        }})

    for _tag in range(0, 1000, 2):
        b = federation.get("RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag), idx=-1)
        print(b)
        c = federation.get("RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag + 1), idx=-1)
        print(c)
        print(c[1].count())
        print(list(c[0].collect()))
