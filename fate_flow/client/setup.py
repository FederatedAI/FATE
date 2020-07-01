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
from setuptools import setup, find_packages

setup(
    name='flow-client',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['Click', 'six', 'ruamel.yaml', 'cachetools',
                      'python-dotenv', 'kazoo', 'requests', 'requests_toolbelt'],
    entry_points='''
        [console_scripts]
        flow=fate_flow.client.scripts.flow:flow_cli
    ''',
)