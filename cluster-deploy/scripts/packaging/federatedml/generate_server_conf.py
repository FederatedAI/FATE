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
import sys
import json
import os


def gen_conf(services_env, server_conf):
    services_env = os.path.realpath(services_env)
    server_conf = os.path.realpath(server_conf)
    services = dict()
    with open(services_env) as fr:
        while True:
            line = fr.readline().strip()
            if not line:
                break
            item, value = line.split('=')
            service_name = item.split('.')[0]
            service_config = '.'.join(item.split('.')[1:])
            services[service_name] = services.get(service_name, {})
            services[service_name][service_config] = value
    with open(server_conf, 'w') as fw:
        json.dump(dict(server=services), fw, indent=4)


if __name__ == '__main__':
    gen_conf(sys.argv[1], sys.argv[2])
