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

from setuptools import setup

packages = ['fate_test']

package_data = {'': ['*']}

install_requires = [
    'click>=7.1.2,<8.0.0',
    'fate_client>=0.1.0,<0.2.0',
    'loguru>=0.5.1,<0.6.0',
    'prettytable>=0.7.2,<0.8.0',
    'requests>=2.24.0,<3.0.0',
    'requests_toolbelt>=0.9.1,<0.10.0',
    'ruamel.yaml>=0.16.10,<0.17.0',
    'sshtunnel>=0.1.5,<0.2.0'
]

entry_points = {
    'console_scripts': ['fate_test = fate_test.cli:cli']
}

setup_kwargs = {
    'name': 'fate-test',
    'version': '0.1.0',
    'description': '',
    'long_description': '',
    'author': 'FederatedAI',
    'author_email': 'contact@FedAI.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://fate.fedai.org/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.6,<4.0',
}

setup(**setup_kwargs)
